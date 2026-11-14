"""
Health and Model Metadata Routes for FastAPI integration backend.
"""

from fastapi import APIRouter
from integration_api.services.existing_model_adapter import (
    ExistingModelAdapter,
    CLASS_NAMES,
    CLASS_DISPLAY_NAMES,
)

router = APIRouter(prefix="/api", tags=["System & Models"])

@router.get("/health")
def health_check():
    adapter = ExistingModelAdapter.get_instance()
    models_status = {
        "custom_cnn": adapter.cnn_model is not None,
        "mobilenetv2_extractor": adapter.mobilenet_extractor is not None,
        "svm_classifier": adapter.svm_model is not None,
        "random_forest": adapter.rf_model is not None,
        "logistic_regression": adapter.lr_model is not None,
        "combined_svm": adapter.combined_svm_model is not None,
    }
    all_ready = all(models_status.values())
    return {
        "status": "healthy" if all_ready else "partial",
        "ready": all_ready,
        "models_status": models_status,
        "supported_classes": [
            {"key": c, "display_name": CLASS_DISPLAY_NAMES[c]}
            for c in CLASS_NAMES
        ],
    }

@router.get("/models")
def list_models():
    return {
        "models": [
            {
                "id": "mobilenetv2_svm",
                "name": "MobileNetV2 + SVM (Best Model)",
                "type": "ImageNet Pretrained Features + Tuned RBF SVM",
                "benchmark_accuracy": "92.50%",
                "feature_dimension": 1280,
                "is_primary": True,
            },
            {
                "id": "combined_hybrid",
                "name": "Combined CNN + MobileNetV2 + SVM",
                "type": "Dual-Stream Feature Fusion (128-d + 1280-d)",
                "benchmark_accuracy": "92.42%",
                "feature_dimension": 1408,
                "is_primary": False,
            },
            {
                "id": "logistic_regression",
                "name": "MobileNetV2 + Logistic Regression",
                "type": "Linear Maximum Margin Baseline",
                "benchmark_accuracy": "89.06%",
                "feature_dimension": 1280,
                "is_primary": False,
            },
            {
                "id": "random_forest",
                "name": "MobileNetV2 + Random Forest",
                "type": "Ensemble Decision Tree Baseline (150 trees)",
                "benchmark_accuracy": "84.71%",
                "feature_dimension": 1280,
                "is_primary": False,
            },
            {
                "id": "custom_cnn",
                "name": "Custom CNN",
                "type": "Standalone 3-Block Convolutional Neural Network",
                "benchmark_accuracy": "64.79%",
                "feature_dimension": "Direct 224x224x3 Input",
                "is_primary": False,
            },
        ]
    }

