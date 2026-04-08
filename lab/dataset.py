from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, split_dir: str | Path, image_size: int = 256) -> None:
        self.split_dir = Path(split_dir)
        self.image_dir = self.split_dir / "images"
        self.mask_dir = self.split_dir / "masks"
        self.image_size = image_size
        self.images = sorted(self.image_dir.glob("*"))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"

        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return image_tensor, mask_tensor

