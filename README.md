# Lab 2: House Segmentation Pipeline

This repository implements the full Lab 2 deliverable set described in `lab_request.txt`:

- Secrets injection with `.env` and `python-dotenv`
- CI/CD with GitHub Actions
- Dataset preparation for house segmentation
- A U-Net segmentation model in PyTorch
- Training, evaluation, and visualization scripts
- A FastAPI inference API
- Docker packaging and optional Docker Compose deployment
- A printable 4-page report source

## Project layout

```text
.
|-- .github/workflows/ci-cd.yml
|-- app/
|   |-- api.py
|   |-- config.pyCI
|   `-- inference.py
|-- lab/
|   |-- dataset.py
|   |-- engine.py
|   |-- evaluate.py
|   |-- metrics.py
|   |-- models.py
|   |-- prepare_dataset.py
|   `-- train.py
|-- report/lab2_report.html
|-- screenshots/README.md
|-- tests/
|   |-- test_api.py
|   `-- test_metrics.py
|-- Dockerfile
|-- docker-compose.yml
|-- main.py
`-- requirements.txt
```

## Secrets injection

1. Copy `.env.example` to `.env`.
2. Update the values in `.env`.
3. Never commit `.env`.

Runtime configuration is loaded in `app/config.py` using `python-dotenv`.

## Dataset preparation

The dataset script supports two modes:

1. Real data mode
   - Place aerial images in `data/raw/images/`
   - Place sidecar JSON annotations in `data/raw/annotations/`
   - Each JSON file must contain a `polygons` array:

```json
{
  "polygons": [
    {
      "label": "house",
      "points": [[10, 20], [90, 20], [90, 80], [10, 80]]
    }
  ]
}
```

2. Synthetic demo mode
   - Generates roof-like rectangles and their masks so the pipeline can be exercised end to end without external data

Run:

```powershell
python -m lab.prepare_dataset --generate-synthetic 120
```

## Training

```powershell
python -m lab.train --epochs 20 --batch-size 8
```

Outputs are written to `artifacts/`:

- `best_model.pt`
- `history.json`
- `metrics.json`
- `training_curves.png`
- `sample_predictions.png`

## Evaluation

```powershell
python -m lab.evaluate --checkpoint artifacts/best_model.pt
```

## API

Run locally:

```powershell
uvicorn main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /config`
- `POST /predict`

## Tests

```powershell
pytest
```

## Docker

Build:

```powershell
docker build -t lab2-house-segmentation .
```

Run:

```powershell
docker run --rm -p 8000:8000 --env-file .env lab2-house-segmentation
```

## CI/CD

The GitHub Actions workflow:

- installs dependencies
- runs tests
- builds the Docker image
- pushes the image to Docker Hub on `main` if `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are configured as repository secrets

## Report

`report/lab2_report.html` is formatted for print-to-PDF with four page sections. Replace the bracketed placeholders with measured values from `artifacts/metrics.json`, `artifacts/training_curves.png`, and `artifacts/sample_predictions.png`, then print it to PDF.

## Submission checklist

- Application code and Dockerfile
- CI/CD workflow
- Dataset preparation and mask generation code
- Model training and evaluation code
- `requirements.txt`
- 4-page PDF exported from `report/lab2_report.html`
- CI/CD run screenshot
- prediction screenshot

