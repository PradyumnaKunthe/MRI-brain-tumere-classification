"""
FastAPI Application Entry Point for MRI Brain Tumor Detection.
Serves as an isolated integration adapter between the existing ML pipelines and the React frontend.
Provides REST endpoints and directly serves the compiled React application.
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from integration_api.routes import health, prediction, metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"

app = FastAPI(
    title="MRI Brain Tumor Detection - Integration API",
    description="Medical Image Classification backend integrating Custom CNN and Pretrained MobileNetV2 + SVM pipelines.",
    version="1.0.0",
)

# Secure CORS configuration:
# In Cloud Run production, React is served directly from the same origin as FastAPI.
# For local development and external frontend proxies, origins can be configured via ALLOWED_ORIGINS.
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(metrics.router)

# Mount outputs directory to serve confusion matrix and training curves
if OUTPUTS_DIR.is_dir():
    app.mount("/api/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Mount compiled React frontend assets if available
if FRONTEND_ASSETS.is_dir():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS)), name="assets")


@app.get("/")
def serve_index():
    index_file = FRONTEND_DIST / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {
        "title": "MRI Brain Tumor Detection Integration API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/api/health",
    }


# Catch-all route for React client-side routing (SPA fallback)
@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    if full_path.startswith("api") or full_path in ("docs", "redoc", "openapi.json"):
        raise HTTPException(status_code=404, detail="API route not found.")
    
    candidate = FRONTEND_DIST / full_path
    if candidate.is_file():
        return FileResponse(candidate)
    
    index_file = FRONTEND_DIST / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    
    raise HTTPException(status_code=404, detail="Page not found.")


if __name__ == "__main__":
    import os
    import uvicorn

    server_port = int(os.environ.get("PORT", 8080))
    print(f"[*] Starting MRI Brain Tumor Detection Integration Server on port {server_port}...")
    uvicorn.run("integration_api.main:app", host="0.0.0.0", port=server_port, reload=False)

