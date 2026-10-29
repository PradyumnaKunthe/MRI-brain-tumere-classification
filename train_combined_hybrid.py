
"""Combined Custom CNN + MobileNetV2 + SVM hybrid model."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset_loader import (
    BASE_DIR,
    OUTPUT_DIR,
    CLASS_NAMES,
    load_manifest,
    preprocess_tensor,
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

MODEL_DIR = BASE_DIR / "models"
FEATURE_DIR = OUTPUT_DIR / "mobilenet_features"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CNN_PATH = MODEL_DIR / "custom_cnn.keras"
MODEL_PATH = MODEL_DIR / "combined_cnn_mobilenet_svm.joblib"

REPORT_PATH = OUTPUT_DIR / "combined_hybrid_report.txt"
CM_PATH = OUTPUT_DIR / "combined_hybrid_confusion_matrix.png"
METRICS_PATH = OUTPUT_DIR / "combined_hybrid_metrics.json"
COMPARISON_PATH = OUTPUT_DIR / "all_model_comparison.csv"

BATCH_SIZE = 16
SEED = 42

tf.keras.utils.set_random_seed(SEED)


# --------------------------------------------------
# 2. LOAD FEATURES AND CHECK ALIGNMENT
# --------------------------------------------------

def load_mobilenet_features(split, paths, labels):
    """Reuse MobileNetV2 features previously extracted."""

    cache_path = FEATURE_DIR / f"{split}_features.npz"

    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing feature cache: {cache_path}\n"
            "Run train_hybrid.py first."
        )

    with np.load(cache_path, allow_pickle=True) as cache:
        features = cache["features"]
        cached_labels = cache["labels"]
        cached_paths = cache["paths"]

    # Prevent incorrectly combining features from different images.
    if not np.array_equal(paths, cached_paths):
        raise ValueError(
            f"Image order/path mismatch in {split} MobileNetV2 cache."
        )

    if not np.array_equal(labels, cached_labels):
        raise ValueError(
            f"Label mismatch in {split} MobileNetV2 cache."
        )

    return features


# --------------------------------------------------
# 3. EXTRACT CUSTOM CNN FEATURES
# --------------------------------------------------

def extract_cnn_features(cnn_feature_model, split):
    """Extract 128-dimensional embeddings from the saved CNN."""

    df = load_manifest(split)

    paths = df["path"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype=np.int32)

    # Build a deterministic dataset with no shuffling.
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        lambda path, label: preprocess_tensor(
            path,
            label,
            model_type="cnn",
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    print(f"\nExtracting custom CNN features: {split.upper()}")

    cnn_features = cnn_feature_model.predict(
        dataset,
        verbose=1,
    )

    if len(cnn_features) != len(labels):
        raise ValueError(
            f"Feature/label count mismatch for {split}."
        )

    # Load cached MobileNetV2 features in the same order.
    mobilenet_features = load_mobilenet_features(
        split,
        paths,
        labels,
    )

    print("CNN features:", cnn_features.shape)
    print("MobileNetV2 features:", mobilenet_features.shape)

    # Fusion: 128 + 1280 = 1408 features.
    combined_features = np.concatenate(
        [cnn_features, mobilenet_features],
        axis=1,
    ).astype(np.float32)

    print("Combined features:", combined_features.shape)

    return combined_features, labels


# --------------------------------------------------
# 4. BUILD COMBINED FEATURE EXTRACTOR
# --------------------------------------------------

def main():
    if not CNN_PATH.is_file():
        raise FileNotFoundError(
            f"Saved CNN not found: {CNN_PATH}\n"
            "Train the custom CNN first."
        )

    # The CNN checkpoint contains the learned CNN weights.
    cnn_model = tf.keras.models.load_model(
        CNN_PATH,
        compile=False,
    )

    # The dense layer has 128 output features.
    cnn_feature_model = tf.keras.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("dense").output,
        name="custom_cnn_feature_extractor",
    )

    # Extract features for the three existing data splits.
    X_train, y_train = extract_cnn_features(
        cnn_feature_model, "train"
    )

    X_val, y_val = extract_cnn_features(
        cnn_feature_model, "validation"
    )

    X_test, y_test = extract_cnn_features(
        cnn_feature_model, "test"
    )

    print("\nFeature fusion completed.")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    # --------------------------------------------------
    # 5. TRAIN ONE SVM ON FUSED FEATURES
    # --------------------------------------------------

    combined_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=False,
            cache_size=512,
            random_state=SEED,
        )),
    ])

    print("\nTraining combined CNN + MobileNetV2 + SVM...")
    combined_svm.fit(X_train, y_train)

    # Validation is used for monitoring.
    val_predictions = combined_svm.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation accuracy: {val_accuracy:.4%}")

    # --------------------------------------------------
    # 6. EVALUATE ON THE HELD-OUT TEST SET
    # --------------------------------------------------

    test_predictions = combined_svm.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    report = classification_report(
        y_test,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("COMBINED CNN + MOBILENETV2 + SVM RESULTS")
    print("=" * 60)
    print(f"Validation accuracy: {val_accuracy:.4%}")
    print(f"Combined test accuracy: {test_accuracy:.4%}")
    print("\nClassification report:\n")
    print(report)

    # Save the combined classifier.
    joblib.dump(
        combined_svm,
        MODEL_PATH,
        compress=3,
    )

    # Save the evaluation report.
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(
            "Combined Custom CNN + MobileNetV2 + SVM\n"
        )
        file.write(
            f"Validation accuracy: {val_accuracy:.4%}\n"
        )
        file.write(
            f"Combined test accuracy: {test_accuracy:.4%}\n\n"
        )
        file.write(report)

    # Save the confusion matrix.
    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Combined CNN + MobileNetV2 + SVM")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=160)
    plt.close(fig)

    # Store metrics for the Streamlit application.
    metrics = {
        "model": "Combined CNN + MobileNetV2 + SVM",
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "test_images": int(len(y_test)),
        "correct_predictions": int(
            np.sum(test_predictions == y_test)
        ),
        "feature_dimensions": int(X_train.shape[1]),
        "class_names": CLASS_NAMES,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSaved classifier: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved confusion matrix: {CM_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")

    # Create one comparison table with the earlier experiments.
    rows = []

    cnn_metrics_path = OUTPUT_DIR / "cnn_metrics.json"
    hybrid_metrics_path = OUTPUT_DIR / "hybrid_model_metrics.json"

    if cnn_metrics_path.exists():
        with cnn_metrics_path.open(encoding="utf-8") as file:
            item = json.load(file)

        rows.append({
            "Model": item["model"],
            "Test accuracy": item["test_accuracy"],
        })

    if hybrid_metrics_path.exists():
        with hybrid_metrics_path.open(encoding="utf-8") as file:
            items = json.load(file)

        for item in items:
            rows.append({
                "Model": item["model"],
                "Test accuracy": item["test_accuracy"],
            })

    rows.append({
        "Model": metrics["model"],
        "Test accuracy": metrics["test_accuracy"],
    })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(COMPARISON_PATH, index=False)

    print("\nALL MODEL TEST ACCURACIES")
    print(comparison.to_string(index=False))
    print(f"\nComparison saved to: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
