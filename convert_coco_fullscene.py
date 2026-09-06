"""Convert PKLot full-scene COCO boxes to YOLO detection format.

The original archive stays outside the application repository.  Images are
linked or copied into a compact train/validation/test workspace and every
parking-space annotation is retained as either empty (0) or occupied (1).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path, help="PKLot folder containing train/valid/test")
parser.add_argument("--output", type=Path, default=Path("data/parking_fullscene"))
parser.add_argument("--train-images", type=int, default=1600)
parser.add_argument("--validation-images", type=int, default=400)
parser.add_argument("--test-images", type=int, default=400)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

rng = random.Random(args.seed)
requested = {"train": args.train_images, "valid": args.validation_images, "test": args.test_images}
output_names = {"train": "train", "valid": "validation", "test": "test"}

for source_name, limit in requested.items():
    source_folder = args.source / source_name
    coco = json.loads((source_folder / "_annotations.coco.json").read_text(encoding="utf-8"))
    images = {item["id"]: item for item in coco["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation["category_id"] in (1, 2):
            annotations[annotation["image_id"]].append(annotation)
    candidates = [image_id for image_id in images if annotations[image_id]]
    rng.shuffle(candidates)
    selected = candidates[: min(limit, len(candidates))]
    destination_name = output_names[source_name]
    image_dir = args.output / "images" / destination_name
    label_dir = args.output / "labels" / destination_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    object_counts = [0, 0]
    for image_id in selected:
        record = images[image_id]
        source_image = source_folder / record["file_name"]
        target_image = image_dir / record["file_name"]
        link_or_copy(source_image, target_image)
        width, height = float(record["width"]), float(record["height"])
        rows = []
        for annotation in annotations[image_id]:
            x, y, box_width, box_height = map(float, annotation["bbox"])
            class_id = annotation["category_id"] - 1
            object_counts[class_id] += 1
            center_x = (x + box_width / 2) / width
            center_y = (y + box_height / 2) / height
            rows.append(
                f"{class_id} {center_x:.7f} {center_y:.7f} "
                f"{box_width / width:.7f} {box_height / height:.7f}"
            )
        (label_dir / f"{target_image.stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(destination_name, "images=", len(selected), "empty=", object_counts[0], "occupied=", object_counts[1])

yaml = f"""path: {args.output.resolve()}
train: images/train
val: images/validation
test: images/test
names:
  0: empty_space
  1: occupied_space
"""
(args.output / "parking_fullscene.yaml").write_text(yaml, encoding="utf-8")
print("Dataset YAML:", args.output / "parking_fullscene.yaml")
