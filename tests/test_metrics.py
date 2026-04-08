import torch

from lab.metrics import dice_score, iou_score


def test_metrics_are_one_for_perfect_overlap() -> None:
    prediction = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    assert dice_score(prediction, target) == 1.0
    assert iou_score(prediction, target) == 1.0


def test_metrics_drop_for_partial_overlap() -> None:
    prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    dice = dice_score(prediction, target)
    iou = iou_score(prediction, target)
    assert 0.0 < dice < 1.0
    assert 0.0 < iou < 1.0

