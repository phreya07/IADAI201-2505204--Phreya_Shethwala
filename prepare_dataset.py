"""Validate, balance, and group-split PKLot crops into 70/15/15 folders."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path, help="PKLot root containing Empty/Occupied folders")
parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed")
parser.add_argument("--per-class", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
rng = random.Random(args.seed)

if args.per_class < 100:
    raise SystemExit("Use at least 100 images per class.")
if args.output.exists() and any(p.suffix.lower() in {".jpg", ".jpeg", ".png"} for p in args.output.rglob("*")):
    raise SystemExit(f"Output already contains images: {args.output}. Use a clean output folder to prevent mixed splits.")


def class_dirs(name: str) -> list[Path]:
    aliases = {"empty": {"empty", "vacant"}, "occupied": {"occupied"}}[name]
    return [p for p in args.source.rglob("*") if p.is_dir() and p.name.casefold() in aliases]


def valid_records(name: str) -> list[tuple[Path, str]]:
    records: list[tuple[Path, str]] = []
    for class_dir in class_dirs(name):
        for path in class_dir.rglob("*"):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                # Patch datasets create many slots from one full image. Keep
                # all spots from that original image in the same split.
                session = path.name.rsplit("_spot_", 1)[0] if "_spot_" in path.name else str(class_dir.parent.resolve())
                records.append((path, session))
            except (UnidentifiedImageError, OSError):
                continue
    return records


records = {name: valid_records(name) for name in ("empty", "occupied")}
for name, items in records.items():
    if len(items) < args.per_class:
        raise SystemExit(f"Need {args.per_class} valid {name} images; found {len(items)}.")

sessions = sorted({session for items in records.values() for _, session in items})
if len(sessions) < 7:
    raise SystemExit("Need at least seven separate capture sessions for a reliable grouped 70/15/15 split.")
rng.shuffle(sessions)
n_train = max(1, round(0.70 * len(sessions)))
n_validation = max(1, round(0.15 * len(sessions)))
session_split = {
    session: "train" if i < n_train else "validation" if i < n_train + n_validation else "test"
    for i, session in enumerate(sessions)
}

quotas = {
    "train": int(0.70 * args.per_class),
    "validation": int(0.15 * args.per_class),
}
quotas["test"] = args.per_class - quotas["train"] - quotas["validation"]

for class_name, items in records.items():
    pools = {split: [] for split in quotas}
    for path, session in items:
        pools[session_split[session]].append(path)
    for split, quota in quotas.items():
        rng.shuffle(pools[split])
        if len(pools[split]) < quota:
            raise SystemExit(
                f"Grouped split has only {len(pools[split])} {class_name} images in {split}; "
                f"need {quota}. Increase --per-class data coverage or try another --seed."
            )
        destination = args.output / split / class_name
        destination.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(pools[split][:quota]):
            shutil.copy2(source, destination / f"{class_name}_{index:05d}{source.suffix.lower()}")
        print(f"{split:10} {class_name:8} {quota}")

print(f"Prepared {args.per_class * 2} balanced images without capture-session leakage.")
