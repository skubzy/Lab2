from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.inference import PredictionService

app = FastAPI(title="Lab 2 House Segmentation API", version="1.0.0")
service = PredictionService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.app_env}


@app.get("/config")
def config() -> dict[str, object]:
    return {
        "app_env": settings.app_env,
        "model_path": settings.model_path,
        "image_size": settings.image_size,
        "data_root": settings.data_root,
        "dockerhub_username_configured": bool(settings.dockerhub_username),
        "dockerhub_token_configured": bool(settings.dockerhub_token),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = service.predict_bytes(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "width": result.width,
        "height": result.height,
        "positive_pixels": result.positive_pixels,
        "mask_png_base64": result.mask_png_base64,
    }
