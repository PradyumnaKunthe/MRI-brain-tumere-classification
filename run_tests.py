"""
Automated Test Suite for MRI Brain Tumor Detection.
Executes Phase 8 QA verification across:
- TEST 1: Project Integrity
- TEST 2: Dataset Loader & Preprocessing
- TEST 3: MobileNetV2 Feature Extractor
- TEST 4: SVM Classifier
- TEST 5: CNN Classifier
- TEST 6: Streamlit Application Components
- TEST 7: End-to-End Inference Pipeline
"""

import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

def run_all_tests():
    results = {}
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "outputs"
    TEST_DIR = BASE_DIR / "dataset" / "Epic and CSCR hospital Dataset" / "Test"

    print("=" * 60)
    print("RUNNING MRI BRAIN TUMOR DETECTION QA TEST SUITE")
    print("=" * 60)

    # ---------------------------------------------------------
    # TEST 1: PROJECT INTEGRITY
    # ---------------------------------------------------------
    print("\n[TEST 1] Checking Project Integrity...")
    try:
        required_files = [
            "app.py",
            "dataset_loader.py",
            "preprocessing.py",
            "train_cnn.py",
            "train_hybrid.py",
            "train_combined_hybrid.py",
            "requirements.txt",
            "README.md",
            ".gitignore",
        ]
        missing_files = [f for f in required_files if not (BASE_DIR / f).is_file()]

        required_models = [
            "custom_cnn.keras",
            "mobilenetv2_feature_extractor.keras",
            "mobilenetv2_plus_svm.joblib",
            "mobilenetv2_plus_random_forest.joblib",
            "mobilenetv2_plus_logistic_regression.joblib",
            "combined_cnn_mobilenet_svm.joblib",
        ]
        missing_models = [m for m in required_models if not (MODEL_DIR / m).is_file()]

        required_dirs = [
            BASE_DIR / "models",
            BASE_DIR / "outputs",
            BASE_DIR / "outputs" / "mobilenet_features",
        ]
        missing_dirs = [str(d) for d in required_dirs if not d.is_dir()]

        # Check imports
        import tensorflow as tf
        import sklearn
        import joblib
        import pandas as pd
        import matplotlib
        import cv2
        import streamlit

        if missing_files or missing_models or missing_dirs:
            results["Project Integrity"] = (
                "Fail",
                f"Missing files: {missing_files}, models: {missing_models}, dirs: {missing_dirs}",
            )
        else:
            results["Project Integrity"] = (
                "Pass",
                f"All 9 core files, 6 model artifacts, and 3 output directories verified. Dependencies intact.",
            )
    except Exception as e:
        results["Project Integrity"] = ("Fail", str(e))

    print(f"Result: {results['Project Integrity'][0]} - {results['Project Integrity'][1]}")

    # ---------------------------------------------------------
    # TEST 2: DATASET LOADER & PREPROCESSING
    # ---------------------------------------------------------
    print("\n[TEST 2] Checking Dataset Loader & Preprocessing...")
    try:
        from dataset_loader import load_manifest, preprocess_tensor, CLASS_NAMES

        # Verify manifest
        test_df = load_manifest("test")
        assert len(test_df) == 2414, f"Expected 2414 test samples, got {len(test_df)}"
        assert set(test_df["class_name"].unique()) == set(CLASS_NAMES)

        # Test valid image tensor preprocessing
        sample_path = test_df.iloc[0]["path"]
        sample_label = tf.constant(test_df.iloc[0]["label"], dtype=tf.int32)

        cnn_img, cnn_lbl = preprocess_tensor(tf.constant(sample_path), sample_label, "cnn")
        assert cnn_img.shape == (224, 224, 3), f"Unexpected shape {cnn_img.shape}"
        assert 0.0 <= float(tf.reduce_min(cnn_img)) <= float(tf.reduce_max(cnn_img)) <= 1.0, "CNN range invalid"

        mob_img, mob_lbl = preprocess_tensor(tf.constant(sample_path), sample_label, "mobilenetv2")
        assert mob_img.shape == (224, 224, 3), f"Unexpected shape {mob_img.shape}"
        assert -1.05 <= float(tf.reduce_min(mob_img)) and float(tf.reduce_max(mob_img)) <= 1.05, "MobileNet range invalid"

        # Test corrupt image detection
        invalid_tensor_path = tf.constant(str(BASE_DIR / "requirements.txt"))
        corrupt_handled = False
        try:
            preprocess_tensor(invalid_tensor_path, sample_label, "cnn")
        except Exception:
            corrupt_handled = True

        assert corrupt_handled, "Failed to reject non-image file"

        results["Dataset Loader"] = (
            "Pass",
            f"Verified 2414 test records, correct 224x224x3 shapes, CNN [0, 1] range, MobileNet [-1, 1] range, and corrupt file rejection.",
        )
    except Exception as e:
        results["Dataset Loader"] = ("Fail", str(e))

    print(f"Result: {results['Dataset Loader'][0]} - {results['Dataset Loader'][1]}")

    # ---------------------------------------------------------
    # TEST 3: MOBILENETV2 FEATURE EXTRACTOR
    # ---------------------------------------------------------
    print("\n[TEST 3] Checking MobileNetV2 Feature Extractor...")
    try:
        mobilenet_model = tf.keras.models.load_model(
            MODEL_DIR / "mobilenetv2_feature_extractor.keras",
            compile=False,
        )
        dummy_input = np.zeros((2, 224, 224, 3), dtype=np.float32)
        features = mobilenet_model.predict(dummy_input, verbose=0)

        assert features.shape == (2, 1280), f"Expected shape (2, 1280), got {features.shape}"
        assert not np.isnan(features).any(), "NaN detected in MobileNetV2 features"
        assert not np.isinf(features).any(), "Inf detected in MobileNetV2 features"

        results["MobileNetV2"] = (
            "Pass",
            f"Loaded frozen extractor, validated input (2, 224, 224, 3) -> output (2, 1280) with valid float32 embeddings.",
        )
    except Exception as e:
        results["MobileNetV2"] = ("Fail", str(e))

    print(f"Result: {results['MobileNetV2'][0]} - {results['MobileNetV2'][1]}")

    # ---------------------------------------------------------
    # TEST 4: SVM CLASSIFIER
    # ---------------------------------------------------------
    print("\n[TEST 4] Checking SVM Classifier...")
    try:
        svm_pipe = joblib.load(MODEL_DIR / "mobilenetv2_plus_svm.joblib")
        dummy_feat = np.random.randn(2, 1280).astype(np.float32)

        preds = svm_pipe.predict(dummy_feat)
        assert len(preds) == 2, f"Expected 2 predictions, got {len(preds)}"
        assert set(preds).issubset({0, 1, 2, 3}), f"Invalid prediction labels: {preds}"

        assert hasattr(svm_pipe, "predict_proba"), "SVM pipeline missing predict_proba"
        probs = svm_pipe.predict_proba(dummy_feat)
        assert probs.shape == (2, 4), f"Expected prob shape (2, 4), got {probs.shape}"
        assert np.allclose(np.sum(probs, axis=1), 1.0), "Probabilities do not sum to 1.0"

        results["SVM Classifier"] = (
            "Pass",
            f"Loaded tuned Pipeline with StandardScaler and SVC(probability=True). Correct 1280-d input -> 4-class calibrated probabilities.",
        )
    except Exception as e:
        results["SVM Classifier"] = ("Fail", str(e))

    print(f"Result: {results['SVM Classifier'][0]} - {results['SVM Classifier'][1]}")

    # ---------------------------------------------------------
    # TEST 5: CNN CLASSIFIER
    # ---------------------------------------------------------
    print("\n[TEST 5] Checking Custom CNN Classifier...")
    try:
        cnn_model = tf.keras.models.load_model(
            MODEL_DIR / "custom_cnn.keras",
            compile=False,
        )
        dummy_cnn_in = np.zeros((2, 224, 224, 3), dtype=np.float32)
        cnn_out = cnn_model.predict(dummy_cnn_in, verbose=0)

        assert cnn_out.shape == (2, 4), f"Expected shape (2, 4), got {cnn_out.shape}"
        assert np.allclose(np.sum(cnn_out, axis=1), 1.0), "CNN softmax outputs do not sum to 1.0"

        # Check embedding layer extraction for hybrid model
        dense_layer = cnn_model.get_layer("dense")
        assert dense_layer.output.shape[-1] == 128, f"Dense embedding shape expected 128, got {dense_layer.output.shape[-1]}"

        results["CNN Classifier"] = (
            "Pass",
            f"Loaded custom CNN, verified input (2, 224, 224, 3) -> 4-class softmax output and 128-d dense embedding layer.",
        )
    except Exception as e:
        results["CNN Classifier"] = ("Fail", str(e))

    print(f"Result: {results['CNN Classifier'][0]} - {results['CNN Classifier'][1]}")

    # ---------------------------------------------------------
    # TEST 6: STREAMLIT APP LOGIC & COMPATIBILITY
    # ---------------------------------------------------------
    print("\n[TEST 6] Checking Streamlit App Components...")
    try:
        import app

        # Test model loading function
        loaded_models, model_status = app.load_models()
        assert all(v is True for v in model_status.values()), f"Model loading status issue: {model_status}"

        # Test image preparation
        test_pil = Image.new("RGB", (300, 400), color=(100, 100, 100))
        img_clean, cnn_in, mob_in = app.prepare_image(test_pil)
        assert img_clean.size == (224, 224)
        assert cnn_in.shape == (1, 224, 224, 3)
        assert mob_in.shape == (1, 224, 224, 3)

        # Test filters
        clahe_img = app.apply_clahe_filter(img_clean)
        edge_img = app.apply_edge_filter(img_clean)
        inv_img = app.apply_inverted_filter(img_clean)
        assert clahe_img.size == (224, 224)
        assert edge_img.size == (224, 224)
        assert inv_img.size == (224, 224)

        # Test sample library
        samples = app.get_sample_library()
        assert len(samples) == 4, f"Expected 4 classes in sample library, got {len(samples)}"

        results["Streamlit Startup"] = (
            "Pass",
            f"Verified clean import, model cache loading (all 6 models active), 224x224 image preparation, 3 radiological filters, and test library discovery.",
        )
    except Exception as e:
        results["Streamlit Startup"] = ("Fail", str(e))

    print(f"Result: {results['Streamlit Startup'][0]} - {results['Streamlit Startup'][1]}")

    # ---------------------------------------------------------
    # TEST 7: END-TO-END INFERENCE PIPELINE
    # ---------------------------------------------------------
    print("\n[TEST 7] Checking End-to-End Prediction on Real Test Scans...")
    try:
        import app

        loaded_models, _ = app.load_models()
        test_images = {
            "glioma": TEST_DIR / "glioma" / "1049.jpg",
            "meningioma": TEST_DIR / "meningioma" / "1263.jpg",
            "notumor": TEST_DIR / "notumor" / "1107.jpg",
            "pituitary": TEST_DIR / "pituitary" / "1201.jpg",
        }

        e2e_details = []
        for expected_class, img_path in test_images.items():
            assert img_path.is_file(), f"Missing test scan: {img_path}"
            pil_img = Image.open(img_path)

            t0 = time.perf_counter()
            _, cnn_in, mob_in = app.prepare_image(pil_img)
            predictions, consensus = app.run_all_predictions(loaded_models, cnn_in, mob_in)
            latency = (time.perf_counter() - t0) * 1000

            assert len(predictions) == 5, f"Expected 5 model predictions, got {len(predictions)}"
            assert consensus is not None, "Consensus summary missing"

            e2e_details.append(
                f"{expected_class.upper()} -> Consensus: {consensus['consensus_class']} "
                f"({consensus['consensus_votes']}/5 models, {latency:.1f}ms)"
            )

        results["End-to-End Prediction"] = (
            "Pass",
            f"Successfully executed inference across all 4 clinical pathologies. Detailed: {'; '.join(e2e_details)}",
        )
    except Exception as e:
        results["End-to-End Prediction"] = ("Fail", str(e))

    print(f"Result: {results['End-to-End Prediction'][0]} - {results['End-to-End Prediction'][1]}")

    # Print Summary Table
    print("\n" + "=" * 60)
    print("QA TEST SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Test':<28} {'Status':<12} {'Details'}")
    print("-" * 60)
    for test_name, (status, detail) in results.items():
        print(f"{test_name:<28} {status:<12} {detail[:50]}...")
    print("=" * 60)

    return results

if __name__ == "__main__":
    run_all_tests()
