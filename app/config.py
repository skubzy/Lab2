from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8000"))
    model_path: str = os.getenv("MODEL_PATH", "artifacts/best_model.pt")
    image_size: int = int(os.getenv("IMAGE_SIZE", "256"))
    inference_threshold: float = float(os.getenv("INFERENCE_THRESHOLD", "0.5"))
    data_root: str = os.getenv("DATA_ROOT", "data")
    dockerhub_username: str = os.getenv("DOCKERHUB_USERNAME", "")
    dockerhub_token: str = os.getenv("DOCKERHUB_TOKEN", "")


settings = Settings()

