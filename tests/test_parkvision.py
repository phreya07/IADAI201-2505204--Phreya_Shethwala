import numpy as np

import automatic_parking
from automatic_parking import VehicleDetection, analyse_automatic, infer_parking_spaces
from core import read_image, summarize


def test_summary_counts():
    detections = [
        VehicleDetection((.10, .20, .22, .42), .91, "car"),
        VehicleDetection((.42, .20, .54, .42), .88, "car"),
    ]
    slots, info = infer_parking_spaces(detections)
    summary = summarize(slots)
    assert summary["occupied"] == 2
    assert summary["available"] >= 1
    assert info["vehicles"] == 2


def test_missing_detector_fails_cleanly(monkeypatch):
    monkeypatch.setattr(automatic_parking, "infer_marked_parking_spaces", lambda *args, **kwargs: ([], {}))
    slots, info = analyse_automatic(np.zeros((240, 320, 3), dtype=np.uint8), None)
    assert slots == []
    assert info["reliability"] == "Detector unavailable"


def test_invalid_upload_is_rejected():
    try:
        read_image(b"not an image")
    except ValueError as exc:
        assert "valid JPG or PNG" in str(exc)
    else:
        raise AssertionError("Invalid input was accepted")
