"""
Prediction Routes for MRI Image Classification.
Executes Custom CNN, MobileNetV2 + SVM, Side-by-Side Comparison, and All-Model Consensus.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from integration_api.services.existing_model_adapter import ExistingModelAdapter

router = APIRouter(prefix="/api/predict", tags=["Prediction"])

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB limit


def validate_image_file(file: UploadFile):
    """Enforce format and size constraints."""
    ext = Path_ext(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )


def Path_ext(filename: str) -> str:
    from pathlib import Path
    return Path(filename).suffix.lower() if filename else ""


@router.post("/cnn")
async def predict_cnn_endpoint(file: UploadFile = File(...)):
    """Inference through Custom CNN."""
    validate_image_file(file)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 15 MB limit.")

    try:
        adapter = ExistingModelAdapter.get_instance()
        result = adapter.predict_cnn(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CNN inference failed: {str(e)}")


@router.post("/hybrid")
async def predict_hybrid_endpoint(file: UploadFile = File(...)):
    """Inference through MobileNetV2 feature extraction + SVM."""
    validate_image_file(file)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 15 MB limit.")

    try:
        adapter = ExistingModelAdapter.get_instance()
        result = adapter.predict_hybrid(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid SVM inference failed: {str(e)}")


@router.post("/compare")
async def predict_compare_endpoint(file: UploadFile = File(...)):
    """Side-by-side comparison of Custom CNN and MobileNetV2 + SVM on identical image."""
    validate_image_file(file)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 15 MB limit.")

    try:
        adapter = ExistingModelAdapter.get_instance()
        result = adapter.predict_compare(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison inference failed: {str(e)}")


@router.post("/all")
async def predict_all_endpoint(file: UploadFile = File(...)):
    """Multi-model inference across all 5 models with ensemble consensus calculation."""
    validate_image_file(file)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File size exceeds 15 MB limit.")

    try:
        adapter = ExistingModelAdapter.get_instance()
        result = adapter.predict_all(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-model inference failed: {str(e)}")

