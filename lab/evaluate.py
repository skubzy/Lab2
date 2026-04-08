from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.config import settings
from lab.dataset import SegmentationDataset
from lab.engine import run_epoch, save_sample_predictions
from lab.models import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net checkpoint.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--image-size", type=int, default=settings.image_size)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SegmentationDataset(Path(args.data_dir) / "test", image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    criterion = nn.BCEWithLogitsLoss()
    metrics = run_epoch(model, loader, criterion, device, optimizer=None)
    save_sample_predictions(model, loader, device, output_dir / "evaluation_predictions.png")

    output = {
        "checkpoint": args.checkpoint,
        "test_loss": metrics["loss"],
        "test_dice": metrics["dice"],
        "test_iou": metrics["iou"],
    }
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
