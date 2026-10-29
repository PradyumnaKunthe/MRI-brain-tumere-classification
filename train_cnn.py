
"""Train and evaluate the standalone MRI Brain Tumor CNN."""

from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from dataset_loader import (
    create_tf_dataset,
    CLASS_NAMES,
)


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "custom_cnn.keras"
HISTORY_PATH = OUTPUT_DIR / "cnn_training_history.png"
CONFUSION_PATH = OUTPUT_DIR / "cnn_confusion_matrix.png"
REPORT_PATH = OUTPUT_DIR / "cnn_classification_report.txt"

IMAGE_SIZE = (224, 224)
NUM_CLASSES = len(CLASS_NAMES)
EPOCHS = 10

SEED = 42
tf.keras.utils.set_random_seed(SEED)


# --------------------------------------------------
# 2. BUILD THE CNN
# --------------------------------------------------

def build_cnn():
    """Build a CNN that learns directly from MRI images."""

    inputs = tf.keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="mri_image",
    )

    # Mild augmentation is active during training only.
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.03)(x)
    x = tf.keras.layers.RandomZoom(0.08)(x)

    # Convolution block 1
    x = tf.keras.layers.Conv2D(
        32, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Convolution block 2
    x = tf.keras.layers.Conv2D(
        64, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Convolution block 3
    x = tf.keras.layers.Conv2D(
        128, 3, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Learn a compact representation of the image.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="custom_mri_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------
# 3. PLOT TRAINING HISTORY
# --------------------------------------------------

def save_training_plot(history):
    """Save accuracy and loss curves."""

    epochs = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        epochs,
        history.history["accuracy"],
        label="Training accuracy",
    )
    axes[0].plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation accuracy",
    )
    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs,
        history.history["loss"],
        label="Training loss",
    )
    axes[1].plot(
        epochs,
        history.history["val_loss"],
        label="Validation loss",
    )
    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(HISTORY_PATH, dpi=160)
    plt.close(fig)

    print(f"Training curves saved to: {HISTORY_PATH}")


# --------------------------------------------------
# 4. EVALUATE ON THE HELD-OUT TEST SET
# --------------------------------------------------

def evaluate_model(model, test_ds):
    """Generate test metrics, a report and a confusion matrix."""

    true_labels = np.concatenate([
        labels.numpy()
        for _, labels in test_ds
    ])

    probabilities = model.predict(
        test_ds,
        verbose=1,
    )

    predicted_labels = np.argmax(probabilities, axis=1)

    test_accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 55)
    print("CUSTOM CNN - TEST RESULTS")
    print("=" * 55)
    print(f"Test accuracy: {test_accuracy:.4%}")
    print("\nClassification report:\n")
    print(report)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(f"Test accuracy: {test_accuracy:.4%}\n\n")
        file.write(report)

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
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
    ax.set_title("Custom CNN - Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=160)
    plt.close(fig)

    print(f"Report saved to: {REPORT_PATH}")
    print(f"Confusion matrix saved to: {CONFUSION_PATH}")

    return test_accuracy


# --------------------------------------------------
# 5. TRAIN
# --------------------------------------------------

def main():
    print("Loading training and validation batches...")

    train_ds = create_tf_dataset(
        "train",
        model_type="cnn",
    )
    val_ds = create_tf_dataset(
        "validation",
        model_type="cnn",
    )
    test_ds = create_tf_dataset(
        "test",
        model_type="cnn",
    )

    print("\nBuilding custom CNN...")
    model = build_cnn()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nStarting CNN training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    save_training_plot(history)

    # Load the best checkpoint, rather than assuming
    # the final training epoch was the best one.
    best_model = tf.keras.models.load_model(MODEL_PATH)

    test_accuracy = evaluate_model(
        best_model,
        test_ds,
    )

    metrics_path = OUTPUT_DIR / "cnn_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model": "Custom CNN",
                "test_accuracy": float(test_accuracy),
                "epochs_completed": len(history.history["loss"]),
                "class_names": CLASS_NAMES,
            },
            file,
            indent=2,
        )

    print("\nCNN training and evaluation complete.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
