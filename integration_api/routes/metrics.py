"""
Metrics, Dataset Statistics, and Clinical Sample Library Routes.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
import pandas as pd
from integration_api.services.existing_model_adapter import (
    OUTPUT_DIR,
    TEST_DATASET_DIR,
    CLASS_NAMES,
    CLASS_DISPLAY_NAMES,
)

router = APIRouter(prefix="/api", tags=["Metrics & Dataset"])

@router.get("/metrics")
def get_model_metrics():
    """Retrieve actual saved model comparison metrics."""
    comp_file = OUTPUT_DIR / "all_model_comparison.csv"
    hybrid_file = OUTPUT_DIR / "hybrid_model_comparison.csv"

    comparison_data = []
    if comp_file.is_file():
        df_comp = pd.read_csv(comp_file)
        for _, row in df_comp.iterrows():
            comparison_data.append({
                "model": str(row["Model"]),
                "test_accuracy": float(row["Test accuracy"]),
                "test_accuracy_pct": f"{float(row['Test accuracy']) * 100:.2f}%",
            })

    detailed_hybrid = []
    if hybrid_file.is_file():
        df_hyb = pd.read_csv(hybrid_file)
        for _, row in df_hyb.iterrows():
            detailed_hybrid.append({
                "model": str(row["model"]),
                "validation_accuracy": float(row["validation_accuracy"]),
                "test_accuracy": float(row["test_accuracy"]),
                "weighted_precision": float(row["weighted_precision"]),
                "weighted_recall": float(row["weighted_recall"]),
                "weighted_f1": float(row["weighted_f1"]),
            })

    return {
        "overall_comparison": comparison_data,
        "hybrid_models": detailed_hybrid,
    }


@router.get("/metrics/report/{model_key}")
def get_classification_report(model_key: str):
    """Retrieve raw text report file for a given model."""
    report_map = {
        "svm": OUTPUT_DIR / "mobilenetv2_plus_svm_report.txt",
        "combined": OUTPUT_DIR / "combined_hybrid_report.txt",
        "rf": OUTPUT_DIR / "mobilenetv2_plus_random_forest_report.txt",
        "lr": OUTPUT_DIR / "mobilenetv2_plus_logistic_regression_report.txt",
        "cnn": OUTPUT_DIR / "cnn_classification_report.txt",
    }
    target = report_map.get(model_key.lower())
    if not target or not target.is_file():
        raise HTTPException(status_code=404, detail="Report file not found.")

    return {
        "model_key": model_key,
        "report": target.read_text(encoding="utf-8"),
    }


@router.get("/dataset/stats")
def get_dataset_stats():
    """Return verified dataset statistics matching the 12,064-sample corpus."""
    records = [
        {"class": "Glioma", "training": 2414, "validation": 604, "test": 755, "total": 3773},
        {"class": "Meningioma", "training": 1747, "validation": 436, "test": 546, "total": 2729},
        {"class": "No Tumor", "training": 1556, "validation": 389, "test": 487, "total": 2432},
        {"class": "Pituitary", "training": 2003, "validation": 501, "test": 626, "total": 3130},
        {"class": "Total", "training": 7720, "validation": 1930, "test": 2414, "total": 12064},
    ]

    return {
        "corpus_total": 12064,
        "training_total": 7720,
        "validation_total": 1930,
        "test_total": 2414,
        "classes": records,
        "normalization_rules": {
            "cnn": "image / 255.0  =>  range [0.0, 1.0]",
            "mobilenetv2": "(image / 127.5) - 1.0  =>  range [-1.0, 1.0]",
            "resolution": "224x224x3 RGB (Lanczos interpolation)",
        },
    }


@router.get("/samples")
def get_sample_images():
    """Return list of available clinical test samples."""
    samples = {}
    if TEST_DATASET_DIR.is_dir():
        for c in CLASS_NAMES:
            c_dir = TEST_DATASET_DIR / c
            if c_dir.is_dir():
                file_list = sorted(list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png")))
                samples[c] = [
                    {
                        "filename": p.name,
                        "class_key": c,
                        "display_name": CLASS_DISPLAY_NAMES[c],
                        "url": f"/api/samples/{c}/{p.name}",
                    }
                    for p in file_list[:8]
                ]
    return {"samples": samples}


@router.get("/samples/{class_name}/{filename}")
def serve_sample_image(class_name: str, filename: str):
    """Serve a test sample image file for preview/inference."""
    if class_name not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail="Invalid class category.")

    file_path = TEST_DATASET_DIR / class_name / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found.")

    return FileResponse(file_path, media_type="image/jpeg")

