from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import torch

from lab.metrics import dice_score, iou_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        if training:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        if training:
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(logits)
        total_loss += float(loss.item())
        total_dice += dice_score(probs, masks)
        total_iou += iou_score(probs, masks)
        batches += 1

    if batches == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}

    return {
        "loss": total_loss / batches,
        "dice": total_dice / batches,
        "iou": total_iou / batches,
    }


def save_history(history: list[dict[str, float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_training_curves(history: list[dict[str, float]], output_path: str | Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_dice = [row["train_dice"] for row in history]
    val_dice = [row["val_dice"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train loss")
    axes[0].plot(epochs, val_loss, label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_dice, label="train dice")
    axes[1].plot(epochs, val_dice, label="val dice")
    axes[1].set_title("Dice Score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_sample_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    output_path: str | Path,
) -> None:
    model.eval()
    batch = next(iter(loader), None)
    if batch is None:
        return

    images, masks = batch
    images = images.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(images))

    max_items = min(3, images.shape[0])
    fig, axes = plt.subplots(max_items, 3, figsize=(9, 3 * max_items))
    if max_items == 1:
        axes = [axes]

    for row in range(max_items):
        image = images[row].detach().cpu().permute(1, 2, 0).numpy()
        truth = masks[row].detach().cpu().squeeze().numpy()
        pred = probs[row].detach().cpu().squeeze().numpy()

        axes[row][0].imshow(image)
        axes[row][0].set_title("Image")
        axes[row][1].imshow(truth, cmap="gray")
        axes[row][1].set_title("Ground Truth")
        axes[row][2].imshow(pred, cmap="magma")
        axes[row][2].set_title("Prediction")

        for axis in axes[row]:
            axis.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
