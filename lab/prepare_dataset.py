from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a house segmentation dataset.")
    parser.add_argument("--raw-dir", default="data/raw", help="Root directory for raw images and annotations.")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for train/val/test splits.")
    parser.add_argument("--image-size", type=int, default=256, help="Square output image size.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generate-synthetic",
        type=int,
        default=0,
        help="If greater than zero, create this many synthetic aerial samples before splitting.",
    )
    return parser.parse_args()


def generate_synthetic_dataset(raw_dir: Path, count: int, image_size: int, seed: int) -> None:
    rng = random.Random(seed)
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        image = Image.new("RGB", (image_size, image_size), color=(58, 112, 66))
        draw = ImageDraw.Draw(image)
        polygons: list[dict[str, object]] = []

        road_y = rng.randint(25, image_size - 45)
        draw.rectangle([(0, road_y), (image_size, road_y + 12)], fill=(88, 88, 88))

        house_count = rng.randint(1, 4)
        for _ in range(house_count):
            x1 = rng.randint(10, image_size - 70)
            y1 = rng.randint(10, image_size - 70)
            width = rng.randint(20, 48)
            height = rng.randint(20, 48)
            x2 = min(image_size - 10, x1 + width)
            y2 = min(image_size - 10, y1 + height)
            roof_color = (rng.randint(120, 200), rng.randint(30, 80), rng.randint(30, 80))
            draw.rectangle([(x1, y1), (x2, y2)], fill=roof_color, outline=(235, 235, 235), width=2)

            polygons.append(
                {
                    "label": "house",
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                }
            )

        stem = f"sample_{index:04d}"
        image.save(images_dir / f"{stem}.png")
        annotation_path = annotations_dir / f"{stem}.json"
        annotation_path.write_text(json.dumps({"polygons": polygons}, indent=2), encoding="utf-8")


def build_mask(image_size: tuple[int, int], polygons: list[dict[str, object]]) -> Image.Image:
    mask = Image.new("L", image_size, color=0)
    draw = ImageDraw.Draw(mask)

    for polygon in polygons:
        if str(polygon.get("label", "")).lower() != "house":
            continue
        points = polygon.get("points", [])
        if len(points) < 3:
            continue
        draw.polygon(points, fill=255)

    return mask


def collect_samples(raw_dir: Path) -> list[tuple[Path, Path]]:
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"
    samples: list[tuple[Path, Path]] = []

    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        annotation_path = annotations_dir / f"{image_path.stem}.json"
        if annotation_path.exists():
            samples.append((image_path, annotation_path))

    return samples


def split_samples(samples: list[tuple[Path, Path]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_split(split_name: str, samples: list[tuple[Path, Path]], output_dir: Path, image_size: int) -> None:
    image_output = output_dir / split_name / "images"
    mask_output = output_dir / split_name / "masks"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    for image_path, annotation_path in samples:
        image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        polygons = annotation.get("polygons", [])
        mask = build_mask(image.size, polygons).resize((image_size, image_size))

        image.save(image_output / f"{image_path.stem}.png")
        mask.save(mask_output / f"{image_path.stem}.png")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if args.generate_synthetic > 0:
        generate_synthetic_dataset(raw_dir, args.generate_synthetic, args.image_size, args.seed)

    samples = collect_samples(raw_dir)
    if not samples:
        raise SystemExit("No labeled samples found. Add images and JSON polygons or use --generate-synthetic.")

    splits = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    for split_name, split_samples_list in splits.items():
        write_split(split_name, split_samples_list, output_dir, args.image_size)

    manifest = {
        "total_samples": len(samples),
        "train_samples": len(splits["train"]),
        "val_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "image_size": args.image_size,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

