"""Train, test and export the two-class full-scene parking detector."""
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
DATASET = ROOT / "configuration" / "parking_dataset.yaml"


def main() -> None:
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATASET), epochs=10, imgsz=320, batch=32,
        patience=3, project=str(ROOT / "results"), name="parking_fullscene",
        fliplr=0.5, seed=42,
    )
    best = YOLO(str(ROOT / "results" / "parking_fullscene" / "weights" / "best.pt"))
    result = best.val(data=str(DATASET), split="test", imgsz=320)
    metrics = {
        "precision": float(result.box.mp), "recall": float(result.box.mr),
        "map50": float(result.box.map50), "map50_95": float(result.box.map),
        "per_class_map50": [float(value) for value in result.box.ap50],
        "classes": ["empty_space", "occupied_space"], "seed": 42,
    }
    (ROOT / "models" / "fullscene_metrics.json").write_text(json.dumps(metrics, indent=2))
    exported = Path(best.export(format="onnx", imgsz=320, dynamic=False, simplify=True))
    (ROOT / "models" / "parking_best.onnx").write_bytes(exported.read_bytes())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
