from pathlib import Path

from automatic_parking import VehicleDetection, infer_marked_parking_spaces, infer_parking_spaces
from core import read_image, summarize


def car(x1, y1, x2, y2, confidence=0.9):
    return VehicleDetection((x1, y1, x2, y2), confidence, "car")


def test_groups_two_rows_and_keeps_detected_cars_occupied():
    detections = [
        car(.10, .10, .20, .34), car(.30, .10, .40, .34),
        car(.10, .60, .20, .84), car(.30, .60, .40, .84),
    ]
    slots, info = infer_parking_spaces(detections)
    assert info["rows"] == 2
    assert info["vehicles"] == 4
    assert sum(s.status == "Occupied" for s in slots) == 4


def test_infers_visible_empty_space_between_cars():
    slots, info = infer_parking_spaces([car(.10, .20, .20, .50), car(.42, .20, .52, .50)])
    assert info["inferred_empty"] >= 1
    assert any(s.status == "Available" for s in slots)


def test_no_vehicle_does_not_invent_capacity():
    slots, info = infer_parking_spaces([])
    assert slots == []
    assert info["reliability"] == "Insufficient evidence"


def test_painted_grid_sample_has_exact_known_result():
    sample = Path(__file__).parents[1] / "samples" / "parkvision_large_12_slot_test.png"
    slots, info = infer_marked_parking_spaces(read_image(sample.read_bytes()))
    summary = summarize(slots)
    assert info["engine"] == "painted-grid"
    assert summary["total"] == 12
    assert summary["occupied"] == 6
    assert summary["available"] == 6
