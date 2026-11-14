"""
Model Adapter Service for MRI Brain Tumor Detection.
Directly interfaces with existing trained model artifacts without modifying any ML code.
Strictly preserves preprocessing formulas, model parameters, and class mappings.
"""

from pathlib import Path
from collections import Counter
import io
import time
import numpy as np
from PIL import Image, ImageOps
import joblib
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
DATASET_DIR = BASE_DIR / "dataset" / "Epic and CSCR hospital Dataset"
TEST_DATASET_DIR = DATASET_DIR / "Test"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

CLASS_DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

CNN_PATH = MODEL_DIR / "custom_cnn.keras"
MOBILENET_PATH = MODEL_DIR / "mobilenetv2_feature_extractor.keras"
COMBINED_SVM_PATH = MODEL_DIR / "combined_cnn_mobilenet_svm.joblib"
SVM_PATH = MODEL_DIR / "mobilenetv2_plus_svm.joblib"
RF_PATH = MODEL_DIR / "mobilenetv2_plus_random_forest.joblib"
LR_PATH = MODEL_DIR / "mobilenetv2_plus_logistic_regression.joblib"


class ExistingModelAdapter:
    _instance = None

    def __init__(self):
        self.cnn_model = None
        self.cnn_feature_model = None
        self.mobilenet_extractor = None
        self.svm_model = None
        self.rf_model = None
        self.lr_model = None
        self.combined_svm_model = None
        self.loaded = False
        self._load_models()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_models(self):
        """Loads all existing trained models once into memory."""
        print("[ExistingModelAdapter] Initializing and loading existing model artifacts...")

        # 1. Custom CNN
        if CNN_PATH.is_file():
            self.cnn_model = tf.keras.models.load_model(CNN_PATH, compile=False)
            self.cnn_feature_model = tf.keras.Model(
                inputs=self.cnn_model.input,
                outputs=self.cnn_model.get_layer("dense").output,
                name="custom_cnn_feature_extractor",
            )
            print(" -> Loaded Custom CNN")

        # 2. MobileNetV2 Feature Extractor
        if MOBILENET_PATH.is_file():
            self.mobilenet_extractor = tf.keras.models.load_model(MOBILENET_PATH, compile=False)
            print(" -> Loaded MobileNetV2 Feature Extractor")

        # 3. MobileNetV2 + SVM (Best Classifier)
        if SVM_PATH.is_file():
            self.svm_model = joblib.load(SVM_PATH)
            print(" -> Loaded MobileNetV2 + SVM")

        # 4. Random Forest
        if RF_PATH.is_file():
            self.rf_model = joblib.load(RF_PATH)
            print(" -> Loaded MobileNetV2 + Random Forest")

        # 5. Logistic Regression
        if LR_PATH.is_file():
            self.lr_model = joblib.load(LR_PATH)
            print(" -> Loaded MobileNetV2 + Logistic Regression")

        # 6. Combined Hybrid SVM
        if COMBINED_SVM_PATH.is_file():
            self.combined_svm_model = joblib.load(COMBINED_SVM_PATH)
            print(" -> Loaded Combined CNN + MobileNetV2 + SVM")

        self.loaded = True

    def preprocess_image(self, image_bytes: bytes):
        """
        Standardize raw image bytes to (224, 224, 3) and generate exact model inputs:
        - cnn_input: range [0.0, 1.0]
        - mobilenet_input: range [-1.0, 1.0]
        """
        pil_image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(pil_image).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)

        cnn_input = np.expand_dims(pixels / 255.0, axis=0)
        mobilenet_input = np.expand_dims(pixels / 127.5 - 1.0, axis=0)

        meta = {
            "original_width": pil_image.width,
            "original_height": pil_image.height,
            "format": pil_image.format or "JPEG",
            "mode": pil_image.mode,
            "standardized_shape": [224, 224, 3],
        }

        return image, cnn_input, mobilenet_input, meta

    def predict_cnn(self, image_bytes: bytes):
        """Execute Custom CNN prediction pipeline."""
        if self.cnn_model is None:
            raise RuntimeError("Custom CNN model artifact is not loaded.")

        _, cnn_input, _, meta = self.preprocess_image(image_bytes)

        t0 = time.perf_counter()
        probs = self.cnn_model.predict(cnn_input, verbose=0)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        probabilities = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }

        return {
            "model": "Custom CNN",
            "type": "Standalone 3-Block CNN",
            "predicted_class": CLASS_DISPLAY_NAMES[pred_class],
            "predicted_class_key": pred_class,
            "predicted_display_name": CLASS_DISPLAY_NAMES[pred_class],
            "confidence": confidence,
            "confidence_pct": f"{confidence * 100:.2f}%",
            "probabilities": probabilities,
            "inference_time_ms": round(latency_ms, 2),
            "timing_ms": {"total_ms": round(latency_ms, 2), "inference_ms": round(latency_ms, 2)},
            "benchmark_accuracy": "64.79%",
            "metadata": meta,
        }

    def predict_hybrid(self, image_bytes: bytes):
        """Execute MobileNetV2 + SVM prediction pipeline."""
        if self.mobilenet_extractor is None or self.svm_model is None:
            raise RuntimeError("MobileNetV2 or SVM model artifact is not loaded.")

        _, _, mobilenet_input, meta = self.preprocess_image(image_bytes)

        t0 = time.perf_counter()
        # 1. Extract 1280-d features
        features = self.mobilenet_extractor.predict(mobilenet_input, verbose=0)
        # 2. Predict with fitted Pipeline
        pred_label = int(self.svm_model.predict(features)[0])
        pred_class = CLASS_NAMES[pred_label]

        # 3. Probabilities
        probabilities = {}
        if hasattr(self.svm_model, "predict_proba"):
            probs = self.svm_model.predict_proba(features)[0]
            for lbl, p in zip(self.svm_model.classes_, probs):
                probabilities[CLASS_NAMES[int(lbl)]] = float(p)
            confidence = probabilities.get(pred_class, 1.0)
        else:
            confidence = 1.0

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "model": "MobileNetV2 + SVM",
            "type": "Pretrained ImageNet Features + Tuned RBF SVM",
            "predicted_class": CLASS_DISPLAY_NAMES[pred_class],
            "predicted_class_key": pred_class,
            "predicted_display_name": CLASS_DISPLAY_NAMES[pred_class],
            "confidence": confidence,
            "confidence_pct": f"{confidence * 100:.2f}%",
            "probabilities": probabilities,
            "inference_time_ms": round(latency_ms, 2),
            "timing_ms": {"total_ms": round(latency_ms, 2), "inference_ms": round(latency_ms, 2)},
            "benchmark_accuracy": "92.50%",
            "metadata": meta,
        }

    def predict_compare(self, image_bytes: bytes):
        """Execute side-by-side comparison of Custom CNN vs MobileNetV2 + SVM."""
        cnn_result = self.predict_cnn(image_bytes)
        svm_result = self.predict_hybrid(image_bytes)

        agreement = (cnn_result["predicted_class_key"] == svm_result["predicted_class_key"])

        return {
            "agreement": agreement,
            "agreement_label": "Models Agree" if agreement else "Models Disagree",
            "consensus": {
                "agreement": agreement,
                "note": (
                    f"Both models independently diagnosed {svm_result['predicted_display_name']}."
                    if agreement
                    else f"Discrepancy: CNN predicted {cnn_result['predicted_display_name']}, whereas MobileNetV2 + SVM diagnosed {svm_result['predicted_display_name']}."
                ),
            },
            "consensus_class": (
                svm_result["predicted_class"] if agreement
                else "Discrepancy (SVM prioritized)"
            ),
            "models": {
                "cnn": cnn_result,
                "mobilenetv2_svm": svm_result,
            },
            "cnn": cnn_result,
            "hybrid_svm": svm_result,
            "metadata": cnn_result["metadata"],
        }

    def predict_all(self, image_bytes: bytes):
        """Execute all 5 models and calculate ensemble consensus."""
        _, cnn_input, mobilenet_input, meta = self.preprocess_image(image_bytes)
        results = []

        # 1. Custom CNN
        if self.cnn_model is not None:
            t0 = time.perf_counter()
            cnn_probs = self.cnn_model.predict(cnn_input, verbose=0)[0]
            cnn_time = (time.perf_counter() - t0) * 1000
            cnn_idx = int(np.argmax(cnn_probs))
            cnn_pred = CLASS_NAMES[cnn_idx]
            results.append({
                "model": "Custom CNN",
                "type": "Standalone 3-Block CNN",
                "benchmark_accuracy": "64.79%",
                "predicted_class": cnn_pred,
                "predicted_display_name": CLASS_DISPLAY_NAMES[cnn_pred],
                "confidence": float(cnn_probs[cnn_idx]),
                "probabilities": {CLASS_NAMES[i]: float(cnn_probs[i]) for i in range(len(CLASS_NAMES))},
                "inference_time_ms": round(cnn_time, 2),
            })

        # Feature Extraction
        mob_feat = None
        cnn_feat = None

        if self.mobilenet_extractor is not None:
            t0 = time.perf_counter()
            mob_feat = self.mobilenet_extractor.predict(mobilenet_input, verbose=0)
            mob_time = (time.perf_counter() - t0) * 1000
        else:
            mob_time = 0.0

        if self.cnn_feature_model is not None:
            cnn_feat = self.cnn_feature_model.predict(cnn_input, verbose=0)

        # 2. Combined Hybrid Model
        if self.combined_svm_model is not None and cnn_feat is not None and mob_feat is not None:
            comb_feat = np.concatenate([cnn_feat, mob_feat], axis=1).astype(np.float32)
            t0 = time.perf_counter()
            comb_pred_label = int(self.combined_svm_model.predict(comb_feat)[0])
            comb_pred = CLASS_NAMES[comb_pred_label]
            comb_probs = {}
            if hasattr(self.combined_svm_model, "decision_function"):
                dec = self.combined_svm_model.decision_function(comb_feat)[0]
                exp_d = np.exp(dec - np.max(dec))
                s_p = exp_d / np.sum(exp_d)
                comb_probs = {CLASS_NAMES[i]: float(s_p[i]) for i in range(len(CLASS_NAMES))}
                comb_conf = float(comb_probs.get(comb_pred, 1.0))
            else:
                comb_conf = 1.0
            comb_time = (time.perf_counter() - t0) * 1000
            results.append({
                "model": "Combined CNN + MobileNetV2 + SVM",
                "type": "Fused Embedding (1408-d)",
                "benchmark_accuracy": "92.42%",
                "predicted_class": comb_pred,
                "predicted_display_name": CLASS_DISPLAY_NAMES[comb_pred],
                "confidence": comb_conf,
                "probabilities": comb_probs,
                "inference_time_ms": round(comb_time, 2),
            })

        # 3. MobileNetV2 + SVM
        if mob_feat is not None and self.svm_model is not None:
            t0 = time.perf_counter()
            svm_label = int(self.svm_model.predict(mob_feat)[0])
            svm_pred = CLASS_NAMES[svm_label]
            svm_probs = {}
            if hasattr(self.svm_model, "predict_proba"):
                probs = self.svm_model.predict_proba(mob_feat)[0]
                svm_probs = {CLASS_NAMES[int(lbl)]: float(p) for lbl, p in zip(self.svm_model.classes_, probs)}
                svm_conf = float(svm_probs.get(svm_pred, 1.0))
            else:
                svm_conf = 1.0
            svm_time = (time.perf_counter() - t0) * 1000 + mob_time
            results.append({
                "model": "MobileNetV2 + SVM",
                "type": "ImageNet Features + Tuned SVM",
                "benchmark_accuracy": "92.50%",
                "predicted_class": svm_pred,
                "predicted_display_name": CLASS_DISPLAY_NAMES[svm_pred],
                "confidence": svm_conf,
                "probabilities": svm_probs,
                "inference_time_ms": round(svm_time, 2),
            })

        # 4. Random Forest
        if mob_feat is not None and self.rf_model is not None:
            t0 = time.perf_counter()
            rf_label = int(self.rf_model.predict(mob_feat)[0])
            rf_pred = CLASS_NAMES[rf_label]
            rf_probs = {}
            if hasattr(self.rf_model, "predict_proba"):
                probs = self.rf_model.predict_proba(mob_feat)[0]
                rf_probs = {CLASS_NAMES[int(lbl)]: float(p) for lbl, p in zip(self.rf_model.classes_, probs)}
                rf_conf = float(rf_probs.get(rf_pred, 1.0))
            else:
                rf_conf = 1.0
            rf_time = (time.perf_counter() - t0) * 1000 + mob_time
            results.append({
                "model": "MobileNetV2 + Random Forest",
                "type": "ImageNet Features + Random Forest",
                "benchmark_accuracy": "84.71%",
                "predicted_class": rf_pred,
                "predicted_display_name": CLASS_DISPLAY_NAMES[rf_pred],
                "confidence": rf_conf,
                "probabilities": rf_probs,
                "inference_time_ms": round(rf_time, 2),
            })

        # 5. Logistic Regression
        if mob_feat is not None and self.lr_model is not None:
            t0 = time.perf_counter()
            lr_label = int(self.lr_model.predict(mob_feat)[0])
            lr_pred = CLASS_NAMES[lr_label]
            lr_probs = {}
            if hasattr(self.lr_model, "predict_proba"):
                probs = self.lr_model.predict_proba(mob_feat)[0]
                lr_probs = {CLASS_NAMES[int(lbl)]: float(p) for lbl, p in zip(self.lr_model.classes_, probs)}
                lr_conf = float(lr_probs.get(lr_pred, 1.0))
            else:
                lr_conf = 1.0
            lr_time = (time.perf_counter() - t0) * 1000 + mob_time
            results.append({
                "model": "MobileNetV2 + Logistic Regression",
                "type": "ImageNet Features + Logistic Regression",
                "benchmark_accuracy": "89.06%",
                "predicted_class": lr_pred,
                "predicted_display_name": CLASS_DISPLAY_NAMES[lr_pred],
                "confidence": lr_conf,
                "probabilities": lr_probs,
                "inference_time_ms": round(lr_time, 2),
            })

        # Compute Consensus
        preds = [r["predicted_class"] for r in results]
        counts = Counter(preds)
        consensus_class, consensus_votes = counts.most_common(1)[0]
        total_votes = len(results)
        consensus_rate = round((consensus_votes / total_votes) * 100, 1)

        prob_accum = {c: 0.0 for c in CLASS_NAMES}
        prob_count = 0
        for r in results:
            if r.get("probabilities"):
                prob_count += 1
                for c in CLASS_NAMES:
                    prob_accum[c] += r["probabilities"].get(c, 0.0)

        mean_probabilities = (
            {c: round(prob_accum[c] / prob_count, 4) for c in CLASS_NAMES}
            if prob_count > 0 else {}
        )

        consensus_summary = {
            "consensus_class": consensus_class,
            "consensus_display_name": CLASS_DISPLAY_NAMES[consensus_class],
            "consensus_votes": consensus_votes,
            "total_votes": total_votes,
            "consensus_rate": consensus_rate,
            "mean_probabilities": mean_probabilities,
            "agreement_status": (
                "Unanimous Consensus" if consensus_votes == total_votes
                else "Majority Agreement" if consensus_rate >= 60
                else "Split / Borderline Diagnosis"
            ),
        }

        return {
            "models": results,
            "ensemble": {
                "majority_class": CLASS_DISPLAY_NAMES[consensus_class],
                "majority_class_key": consensus_class,
                "consensus_status": consensus_summary["agreement_status"],
                "majority_confidence_pct": f"{consensus_rate:.1f}",
                "average_probabilities": mean_probabilities,
            },
            "consensus": consensus_summary,
            "metadata": meta,
        }

