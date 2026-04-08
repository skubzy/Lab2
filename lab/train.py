from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from app.config import settings
from lab.dataset import SegmentationDataset
from lab.engine import run_epoch, save_history, save_sample_predictions, save_training_curves
from lab.models import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net for house segmentation.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=settings.image_size)
    return parser.parse_args()


def evaluate_split(model: UNet, loader: DataLoader, criterion: nn.Module, device: str) -> dict[str, float]:
    return run_epoch(model, loader, criterion, device, optimizer=None)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SegmentationDataset(Path(args.data_dir) / "train", image_size=args.image_size)
    val_dataset = SegmentationDataset(Path(args.data_dir) / "val", image_size=args.image_size)
    test_dataset = SegmentationDataset(Path(args.data_dir) / "test", image_size=args.image_size)

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise SystemExit("Train/val/test splits are incomplete. Run the dataset preparation step first.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float]] = []
    best_val_dice = float("-inf")
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = evaluate_split(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        history.append(row)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

    save_history(history, output_dir / "history.json")
    save_training_curves(history, output_dir / "training_curves.png")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_split(model, test_loader, criterion, device)
    save_sample_predictions(model, test_loader, device, output_dir / "sample_predictions.png")

    metrics = {
        "best_epoch": checkpoint.get("epoch"),
        "test_loss": test_metrics["loss"],
        "test_dice": test_metrics["dice"],
        "test_iou": test_metrics["iou"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

