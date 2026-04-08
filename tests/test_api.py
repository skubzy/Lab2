import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

import app.api as api_module


class StubPredictionService:
    def predict_bytes(self, image_bytes: bytes):
        return type(
            "StubResult",
            (),
            {
                "mask_png_base64": base64.b64encode(b"mask").decode("utf-8"),
                "positive_pixels": 12,
                "height": 32,
                "width": 32,
            },
        )()


def create_png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint() -> None:
    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "service", StubPredictionService())
    client = TestClient(api_module.app)
    response = client.post(
        "/predict",
        files={"file": ("test.png", create_png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["positive_pixels"] == 12
    assert payload["width"] == 32
    assert payload["height"] == 32

