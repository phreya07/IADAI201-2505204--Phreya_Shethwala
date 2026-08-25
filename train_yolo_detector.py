"""Train and export a custom parking detector from YOLO-labelled full scenes."""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
DATASET = ROOT / "configuration" / "parking_dataset.yaml"


def main() -> None:
    model = YOLO("yolo11n-obb.pt")
    model.train(
        data=str(DATASET), epochs=40, imgsz=1024, batch=8,
        patience=8, project=str(ROOT / "results"), name="parking_yolo",
        degrees=8, brightness=0.20, fliplr=0.5,
    )
    best = YOLO(str(ROOT / "results" / "parking_yolo" / "weights" / "best.pt"))
    best.val(data=str(DATASET), split="test")
    best.export(format="onnx", imgsz=1024, dynamic=False, simplify=True)
    print("Copy the exported best.onnx to models/parking_best.onnx after reviewing test mAP.")


if __name__ == "__main__":
    main()
