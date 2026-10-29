
"""
MRI Brain Tumor Detection - Image Preprocessing Pipeline

Expected dataset structure:
dataset/
├── Train/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/

This script:
1. Checks the dataset structure.
2. Detects and skips unreadable image files.
3. Creates stratified training and validation splits.
4. Keeps the original Test folder separate.
5. Saves dataset manifests and split CSV files.
6. Provides reusable image preprocessing functions.
"""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset" / "Epic and CSCR hospital Dataset"
OUTPUT_DIR = BASE_DIR / "outputs"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

CLASS_TO_LABEL = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

IMAGE_SIZE = (224, 224)
VALIDATION_SIZE = 0.20
RANDOM_SEED = 42

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# --------------------------------------------------
# 2. IMAGE VALIDATION
# --------------------------------------------------

def is_valid_image(image_path: Path) -> bool:
    """Return True if the file can be recognized as an image."""

    try:
        with Image.open(image_path) as image:
            image.verify()
        return True

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return False


# --------------------------------------------------
# 3. DATASET SCANNING
# --------------------------------------------------

def scan_split(split_name: str) -> list[dict]:
    """
    Scan Train or Test and return metadata for valid images.

    Labels are assigned from the parent class folder.
    Images are not loaded into memory all at once.
    """

    split_dir = DATASET_DIR / split_name

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing dataset folder: {split_dir}\n"
            "Check that your dataset contains Train and Test folders."
        )

    records = []
    invalid_files = []

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}\n"
                f"Expected classes: {', '.join(CLASS_NAMES)}"
            )

        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        valid_count = 0

        for image_path in image_paths:
            if not is_valid_image(image_path):
                invalid_files.append(image_path)
                continue

            records.append({
                "path": str(image_path.resolve()),
                "class_name": class_name,
                "label": CLASS_TO_LABEL[class_name],
                "original_split": split_name,
            })

            valid_count += 1

        print(
            f"  {class_name:<12} "
            f"{valid_count:>6} valid images"
        )

    if invalid_files:
        print(
            f"\nWarning: skipped {len(invalid_files)} "
            f"unreadable image file(s) in {split_name}."
        )

        for path in invalid_files[:10]:
            print(f"  Invalid: {path}")

        if len(invalid_files) > 10:
            print("  Additional invalid files omitted.")

    if not records:
        raise ValueError(
            f"No valid images found in {split_dir}."
        )

    return records


# --------------------------------------------------
# 4. CSV EXPORT
# --------------------------------------------------

def save_csv(records: list[dict], output_path: Path) -> None:
    """Save image metadata to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "class_name",
        "label",
        "original_split",
        "partition",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# 5. STRATIFIED TRAIN / VALIDATION SPLIT
# --------------------------------------------------

def create_dataset_splits() -> None:
    """
    Create training, validation and test manifests.

    Training images are divided into 80% train and 20% validation.
    The existing Test folder is kept separate and unchanged.
    """

    print("\nScanning training dataset...")
    training_pool = scan_split("Train")

    print("\nScanning test dataset...")
    test_records = scan_split("Test")

    # Ensure every class has enough examples to stratify.
    class_counts = Counter(
        record["label"] for record in training_pool
    )

    missing_classes = [
        name
        for name, label in CLASS_TO_LABEL.items()
        if class_counts[label] < 2
    ]

    if missing_classes:
        raise ValueError(
            "At least two valid training images per class "
            "are needed for a stratified split. Problem classes: "
            + ", ".join(missing_classes)
        )

    train_records, validation_records = train_test_split(
        training_pool,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=[record["label"] for record in training_pool],
    )

    # Mark each record's final role.
    train_records = [
        {**record, "partition": "train"}
        for record in train_records
    ]

    validation_records = [
        {**record, "partition": "validation"}
        for record in validation_records
    ]

    test_records = [
        {**record, "partition": "test"}
        for record in test_records
    ]

    # Save manifests.
    save_csv(train_records, OUTPUT_DIR / "train.csv")
    save_csv(
        validation_records,
        OUTPUT_DIR / "validation.csv",
    )
    save_csv(test_records, OUTPUT_DIR / "test.csv")

    manifest = (
        train_records
        + validation_records
        + test_records
    )

    save_csv(manifest, OUTPUT_DIR / "dataset_manifest.csv")

    print("\nDataset preparation complete.")
    print(f"Training images:   {len(train_records)}")
    print(f"Validation images: {len(validation_records)}")
    print(f"Test images:       {len(test_records)}")
    print(f"Total images:      {len(manifest)}")

    print("\nLabel mapping:")
    for class_name, label in CLASS_TO_LABEL.items():
        print(f"  {label}: {class_name}")

    print("\nNote: CSV files contain image paths and labels.")
    print("Images are not duplicated or saved as resized copies.")


# --------------------------------------------------
# 6. SINGLE-IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(
    image_path: str | Path,
    model_type: str = "cnn",
) -> np.ndarray:
    """
    Read one image and return a normalized float32 array.

    model_type="cnn":
        Pixel values scaled to [0, 1].

    model_type="mobilenetv2":
        Pixel values scaled to [-1, 1], as expected by
        TensorFlow Keras MobileNetV2 preprocess_input.

    Returns:
        Array with shape (224, 224, 3).
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    if model_type not in {"cnn", "mobilenetv2"}:
        raise ValueError(
            "model_type must be 'cnn' or 'mobilenetv2'."
        )

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS,
            )
            array = np.asarray(
                image,
                dtype=np.float32,
            )

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            f"Could not process image: {image_path}"
        ) from error

    if model_type == "cnn":
        array = array / 255.0

    elif model_type == "mobilenetv2":
        array = (array / 127.5) - 1.0

    if array.shape != (224, 224, 3):
        raise ValueError(
            f"Unexpected image shape: {array.shape}"
        )

    return array


# --------------------------------------------------
# 7. RUN PIPELINE
# --------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("MRI BRAIN TUMOR - DATA PREPROCESSING")
    print("=" * 55)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Outputs: {OUTPUT_DIR}")

    create_dataset_splits()
