from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.config import settings
from lab.models import UNet


@dataclass
class PredictionResult:
    mask_png_base64: str
    positive_pixels: int
    height: int
    width: int


class PredictionService:
    def __init__(self, model_path: str | None = None, image_size: int | None = None) -> None:
        self.model_path = model_path or settings.model_path
        self.image_size = image_size or settings.image_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: UNet | None = None

    def _build_model(self) -> UNet:
        model = UNet(in_channels=3, out_channels=1)
        checkpoint_path = Path(self.model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Train the model first.")

        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    @property
    def model(self) -> UNet:
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def predict_bytes(self, image_bytes: bytes, threshold: float | None = None) -> PredictionResult:
        threshold = settings.inference_threshold if threshold is None else threshold
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_width, original_height = image.size

        resized = image.resize((self.image_size, self.image_size))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary_mask = (probs >= threshold).astype(np.uint8) * 255
        mask_image = Image.fromarray(binary_mask).resize((original_width, original_height))

        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        positive_pixels = int(np.count_nonzero(np.asarray(mask_image)))

        return PredictionResult(
            mask_png_base64=encoded,
            positive_pixels=positive_pixels,
            height=original_height,
            width=original_width,
        )
