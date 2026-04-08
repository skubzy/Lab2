from __future__ import annotations

import torch


def _binarize(prediction: torch.Tensor, threshold: float) -> torch.Tensor:
    if prediction.dtype.is_floating_point:
        return (prediction >= threshold).float()
    return prediction.float()


def dice_score(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred = _binarize(prediction, threshold)
    target = target.float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return float((2.0 * intersection + eps) / (union + eps))


def iou_score(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred = _binarize(prediction, threshold)
    target = target.float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    return float((intersection + eps) / (union + eps))
