import json
from pathlib import Path

from core import read_image, summarize, SlotPrediction


def test_model_and_metrics_are_bundled():
    root = Path(__file__).parents[1]
    assert (root / "models" / "parking_best.onnx").stat().st_size > 1_000_000
    metrics = json.loads((root / "models" / "fullscene_metrics.json").read_text())
    assert metrics["test_images"] == 400
    assert metrics["test_instances"] == 23248
    assert metrics["map50"] > 0.95


def test_summary_counts():
    items = [
        SlotPrediction("P001", "Occupied", .9, .9, [[0,0],[.1,0],[.1,.1],[0,.1]]),
        SlotPrediction("P002", "Available", .8, .2, [[.1,0],[.2,0],[.2,.1],[.1,.1]]),
    ]
    result = summarize(items)
    assert result["total"] == 2
    assert result["occupied"] == 1
    assert result["available"] == 1


def test_invalid_upload_is_rejected():
    try:
        read_image(b"not an image")
    except ValueError as exc:
        assert "valid JPG or PNG" in str(exc)
    else:
        raise AssertionError("Invalid input was accepted")
