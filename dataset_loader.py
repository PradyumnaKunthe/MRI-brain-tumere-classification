
"""Batch data loader for MRI Brain Tumor Detection."""

from pathlib import Path

import pandas as pd
import tensorflow as tf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
SEED = 42

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


# --------------------------------------------------
# Load and validate CSV manifests
# --------------------------------------------------

def load_manifest(split: str) -> pd.DataFrame:
    """Load a CSV manifest and validate its paths and labels."""

    if split not in {"train", "validation", "test"}:
        raise ValueError(
            "split must be train, validation, or test."
        )

    csv_path = OUTPUT_DIR / f"{split}.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing manifest: {csv_path}\n"
            "Run preprocessing.py first."
        )

    df = pd.read_csv(csv_path)

    required_columns = {"path", "class_name", "label"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{csv_path.name} is missing columns: {missing}"
        )

    if df.empty:
        raise ValueError(f"{csv_path.name} contains no images.")

    if df["path"].isna().any():
        raise ValueError(f"{csv_path.name} contains empty paths.")

    if df["label"].isna().any():
        raise ValueError(f"{csv_path.name} contains empty labels.")

    df["label"] = df["label"].astype(int)

    if not df["label"].between(0, NUM_CLASSES - 1).all():
        raise ValueError("CSV contains an invalid numeric label.")

    expected_labels = df["class_name"].map(
        {name: i for i, name in enumerate(CLASS_NAMES)}
    )

    if expected_labels.isna().any():
        raise ValueError("CSV contains an unexpected class name.")

    if not (expected_labels.astype(int) == df["label"]).all():
        raise ValueError(
            "Class names and numeric labels do not match."
        )

    missing_paths = [
        str(path)
        for path in df["path"]
        if not Path(path).is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} image path(s) no longer exist. "
            f"First missing path: {missing_paths[0]}"
        )

    return df


# --------------------------------------------------
# Decode and preprocess one image
# --------------------------------------------------

def preprocess_tensor(
    image_path: tf.Tensor,
    label: tf.Tensor,
    model_type: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode, resize and normalize one image."""

    image_bytes = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    image = tf.cast(image, tf.float32)

    if model_type == "cnn":
        # CNN input range: [0, 1]
        image = image / 255.0

    elif model_type == "mobilenetv2":
        # MobileNetV2 input range: [-1, 1]
        image = (image / 127.5) - 1.0

    else:
        raise ValueError(
            "model_type must be 'cnn' or 'mobilenetv2'."
        )

    image = tf.ensure_shape(image, [224, 224, 3])

    return image, label


# --------------------------------------------------
# Build a batched TensorFlow dataset
# --------------------------------------------------

def create_tf_dataset(
    split: str,
    model_type: str = "cnn",
) -> tf.data.Dataset:
    """Build a memory-efficient, batched dataset."""

    if model_type not in {"cnn", "mobilenetv2"}:
        raise ValueError(
            "model_type must be 'cnn' or 'mobilenetv2'."
        )

    df = load_manifest(split)

    paths = df["path"].astype(str).to_numpy()
    labels = df["label"].to_numpy(dtype="int32")

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if split == "train":
        dataset = dataset.shuffle(
            buffer_size=min(len(df), 2048),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda path, label: preprocess_tensor(
            path, label, model_type
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# --------------------------------------------------
# Load all three splits
# --------------------------------------------------

def get_datasets(model_type: str = "cnn"):
    """Return train, validation and test datasets."""

    train_ds = create_tf_dataset("train", model_type)
    val_ds = create_tf_dataset("validation", model_type)
    test_ds = create_tf_dataset("test", model_type)

    return train_ds, val_ds, test_ds


# --------------------------------------------------
# Test the loader
# --------------------------------------------------

if __name__ == "__main__":
    print("Checking dataset manifests...")

    for split in ["train", "validation", "test"]:
        df = load_manifest(split)
        print(f"\n{split.upper()}: {len(df)} images")
        print(df["class_name"].value_counts().to_string())

    print("\nTesting CNN data loader...")

    train_ds = create_tf_dataset(
        "train",
        model_type="cnn",
    )

    images, labels = next(iter(train_ds))

    print("\nFirst batch loaded successfully!")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Image data type:", images.dtype)
    print(
        "Pixel range:",
        float(tf.reduce_min(images).numpy()),
        "to",
        float(tf.reduce_max(images).numpy()),
    )

    print("\nDataset loader is working.")
