
"""
MRI Brain Tumor Detection
MobileNetV2 feature extraction + SVM, Random Forest,
and Logistic Regression classifiers.
"""

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from dataset_loader import (
    BASE_DIR,
    OUTPUT_DIR,
    CLASS_NAMES,
    load_manifest,
    preprocess_tensor,
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIR = OUTPUT_DIR / "mobilenet_features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_MODEL_PATH = (
    MODEL_DIR / "mobilenetv2_feature_extractor.keras"
)

BATCH_SIZE = 16
RANDOM_SEED = 42

tf.keras.utils.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# --------------------------------------------------
# 2. CREATE OR LOAD MOBILENETV2
# --------------------------------------------------

def get_feature_extractor():
    """
    Load a saved MobileNetV2 feature extractor or initialize
    one with ImageNet pretrained weights.

    Input pixels must already be scaled to [-1, 1].
    Output: 1280 feature values per image.
    """

    if FEATURE_MODEL_PATH.exists():
        print("Loading saved MobileNetV2 feature extractor...")
        return tf.keras.models.load_model(
            FEATURE_MODEL_PATH,
            compile=False,
        )

    print("\nInitializing pretrained MobileNetV2...")
    print("ImageNet weights may need to download on first use.")

    model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    model.trainable = False
    model.save(FEATURE_MODEL_PATH)

    print(f"Feature extractor saved: {FEATURE_MODEL_PATH}")

    return model


# --------------------------------------------------
# 3. BUILD AN ORDERED DATASET
# --------------------------------------------------

def build_feature_dataset(df: pd.DataFrame):
    """
    Build a dataset in manifest order.

    No shuffling is applied here so feature rows always
    correspond to the saved labels and paths.
    """

    paths = df["path"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        lambda path, label: preprocess_tensor(
            path,
            label,
            model_type="mobilenetv2",
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, paths, labels


# --------------------------------------------------
# 4. EXTRACT AND CACHE FEATURES
# --------------------------------------------------

def extract_features(
    feature_extractor,
    split: str,
):
    """
    Extract MobileNetV2 features for a dataset split.

    Cached features are reused if they still match the
    CSV manifest in the same order.
    """

    df = load_manifest(split)

    cache_path = FEATURE_DIR / f"{split}_features.npz"

    paths = df["path"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype=np.int32)

    if cache_path.exists():
        print(f"\nChecking cached features: {cache_path}")

        with np.load(cache_path, allow_pickle=False) as cached:
            cached_features = cached["features"]
            cached_labels = cached["labels"]
            cached_paths = cached["paths"]

        if (
            np.array_equal(cached_paths, paths)
            and np.array_equal(cached_labels, labels)
            and len(cached_features) == len(labels)
        ):
            print(f"Reusing {split} features.")
            return cached_features, cached_labels

        print("Cache doesn't match the manifest. Re-extracting.")

    print(f"\nExtracting MobileNetV2 features: {split.upper()}")

    dataset, paths, labels = build_feature_dataset(df)

    features = feature_extractor.predict(
        dataset,
        verbose=1,
    )

    features = np.asarray(features, dtype=np.float32)

    if len(features) != len(labels):
        raise ValueError(
            f"Feature and label counts differ for {split}: "
            f"{len(features)} features vs {len(labels)} labels."
        )

    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        paths=paths,
    )

    print("Feature shape:", features.shape)
    print(f"Cached features: {cache_path}")

    return features, labels


# --------------------------------------------------
# 5. EVALUATION AND REPORTS
# --------------------------------------------------

def evaluate_classifier(
    name,
    classifier,
    X_val,
    y_val,
    X_test,
    y_test,
):
    """
    Evaluate on validation data and held-out test data.

    The test set is used for final reporting, not for
    selecting hyperparameters.
    """

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {name}")
    print(f"{'=' * 60}")

    classifier.fit(X_train, y_train)

    # Validation score for monitoring only.
    val_predictions = classifier.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    print(f"Validation accuracy: {val_accuracy:.4%}")

    # Final held-out test evaluation.
    test_predictions = classifier.predict(X_test)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            test_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    report = classification_report(
        y_test,
        test_predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print(f"\nTest accuracy: {test_accuracy:.4%}")
    print("\nClassification report:")
    print(report)

    # Save classifier.
    safe_name = name.lower().replace(" ", "_").replace("+", "plus")
    model_path = MODEL_DIR / f"{safe_name}.joblib"

    joblib.dump(
        classifier,
        model_path,
        compress=3,
    )

    # Save text report.
    report_path = OUTPUT_DIR / f"{safe_name}_report.txt"

    with report_path.open("w", encoding="utf-8") as file:
        file.write(f"Model: {name}\n")
        file.write(f"Validation accuracy: {val_accuracy:.4%}\n")
        file.write(f"Test accuracy: {test_accuracy:.4%}\n\n")
        file.write(report)

    # Save confusion matrix.
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

    ax.set_title(f"{name} - Test Confusion Matrix")
    fig.tight_layout()

    cm_path = OUTPUT_DIR / f"{safe_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)

    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")

    return {
        "model": name,
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
    }


# --------------------------------------------------
# 6. MAIN TRAINING PIPELINE
# --------------------------------------------------

def main():
    global X_train, y_train

    print("=" * 60)
    print("MOBILENETV2 HYBRID MODEL TRAINING")
    print("=" * 60)

    # Extract the feature representations.
    feature_extractor = get_feature_extractor()

    X_train, y_train = extract_features(
        feature_extractor,
        "train",
    )

    X_val, y_val = extract_features(
        feature_extractor,
        "validation",
    )

    X_test, y_test = extract_features(
        feature_extractor,
        "test",
    )

    print("\nFeature extraction finished.")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    # 1. Support Vector Machine.
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            cache_size=512,
            random_state=RANDOM_SEED,
        )),
    ])

    # 2. Random Forest.
    random_forest_model = RandomForestClassifier(
        n_estimators=150,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # 3. Logistic Regression.
    logistic_regression_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        )),
    ])

    results = []

    results.append(
        evaluate_classifier(
            "MobileNetV2 + SVM",
            svm_model,
            X_val,
            y_val,
            X_test,
            y_test,
        )
    )

    results.append(
        evaluate_classifier(
            "MobileNetV2 + Random Forest",
            random_forest_model,
            X_val,
            y_val,
            X_test,
            y_test,
        )
    )

    results.append(
        evaluate_classifier(
            "MobileNetV2 + Logistic Regression",
            logistic_regression_model,
            X_val,
            y_val,
            X_test,
            y_test,
        )
    )

    # Save a comparison table.
    comparison = pd.DataFrame(results)

    comparison_path = OUTPUT_DIR / "hybrid_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    # Save the metrics as JSON for the application.
    metrics_path = OUTPUT_DIR / "hybrid_model_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n" + "=" * 60)
    print("HYBRID MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False))

    print(f"\nComparison CSV: {comparison_path}")
    print(f"Metrics JSON: {metrics_path}")
    print("\nAll three hybrid classifiers have finished.")


if __name__ == "__main__":
    main()
