"""Automatic vehicle detection and parking-space inference.

The detector uses a COCO-pretrained Ultralytics YOLO model to locate road
vehicles.  Parking rows and empty gaps are inferred from the detected vehicle
centres, so the upload does not require a hand-written slot map.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core import SlotPrediction


COCO_VEHICLE_CLASSES = {2, 3, 5, 7}
DOTA_VEHICLE_CLASSES = {9, 10}  # large-vehicle, small-vehicle in aerial DOTA


@dataclass(frozen=True)
class VehicleDetection:
    box: tuple[float, float, float, float]
    confidence: float
    class_name: str


def _projection_lines(mask: np.ndarray, vertical: bool) -> list[int]:
    """Return clustered bright-line coordinates from a binary paint mask."""
    length = mask.shape[0] if vertical else mask.shape[1]
    projection = mask.sum(axis=0 if vertical else 1) / 255.0
    kernel = max(3, int((mask.shape[1] if vertical else mask.shape[0]) * 0.004))
    smoothed = np.convolve(projection, np.ones(kernel) / kernel, mode="same")
    indices = np.flatnonzero(smoothed > max(0.34 * length, np.percentile(smoothed, 90)))
    groups: list[list[int]] = []
    for value in indices.tolist():
        if not groups or value - groups[-1][-1] > max(4, kernel * 2):
            groups.append([value])
        else:
            groups[-1].append(value)
    return [round(float(np.mean(group))) for group in groups if len(group) >= 2]


def _line_pairs(lines: list[int], dimension: int) -> list[tuple[int, int]]:
    """Merge double painted edges and convert boundaries to regular bays."""
    lines = [value for value in sorted(lines) if 0.03 * dimension < value < 0.97 * dimension]
    if len(lines) < 2:
        return []
    raw_gaps = np.diff(lines)
    large_gap = float(np.percentile(raw_gaps, 70))
    merge_distance = max(5.0, min(0.055 * dimension, 0.28 * large_gap))
    groups: list[list[int]] = [[lines[0]]]
    for value in lines[1:]:
        if value - groups[-1][-1] <= merge_distance:
            groups[-1].append(value)
        else:
            groups.append([value])
    boundaries = [round(float(np.mean(group))) for group in groups]
    # Remove a bright car/body edge that divides one otherwise regular bay.
    changed = True
    while changed and len(boundaries) >= 4:
        changed = False
        gaps = np.diff(boundaries)
        typical = float(np.percentile(gaps, 70))
        for index in range(1, len(boundaries) - 1):
            left_gap, right_gap = boundaries[index] - boundaries[index - 1], boundaries[index + 1] - boundaries[index]
            if left_gap < 0.68 * typical and right_gap < 0.68 * typical and 0.72 * typical < left_gap + right_gap < 1.32 * typical:
                boundaries.pop(index)
                changed = True
                break
    pairs = list(zip(boundaries, boundaries[1:]))
    widths = [right - left for left, right in pairs]
    if not widths:
        return []
    median_width = float(np.median(widths))
    return [(left, right) for left, right in pairs if 0.55 * median_width <= right - left <= 1.65 * median_width]


def infer_marked_parking_spaces(image: np.ndarray, occupancy_model: Any | None = None) -> tuple[list[SlotPrediction], dict[str, Any]]:
    """Detect a regular painted parking grid and classify its bays visually."""
    import cv2

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    paint = cv2.inRange(hsv, (0, 0, 135), (180, 72, 255))
    paint = cv2.morphologyEx(paint, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    y_pairs = _line_pairs(_projection_lines(paint, False), height)
    global_x_pairs = _line_pairs(_projection_lines(paint, True), width)
    if len(y_pairs) < 1:
        return [], {"rows": 0, "vehicles": 0, "inferred_empty": 0, "reliability": "Insufficient evidence", "engine": "painted-grid"}

    measurements: list[float] = []
    crops: list[np.ndarray] = []
    geometry: list[tuple[int, int, int, int]] = []
    row_columns: list[int] = []
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 135)
    for top, bottom in y_pairs:
        row_height = bottom - top
        lines = cv2.HoughLinesP(
            edges[top:bottom], 1, np.pi / 360, threshold=max(22, row_height // 3),
            minLineLength=max(28, round(row_height * .62)), maxLineGap=max(8, row_height // 10),
        )
        x_lines: list[int] = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
                if dy > 3 * max(1, dx) and dy >= .62 * row_height:
                    x_lines.append(round((int(x1) + int(x2)) / 2))
        row_pairs = global_x_pairs if len(global_x_pairs) >= 2 else _line_pairs(x_lines, width)
        row_columns.append(len(row_pairs))
        for left, right in row_pairs:
            pad_x, pad_y = round((right - left) * .20), round((bottom - top) * .12)
            center = image[top + pad_y:bottom - pad_y, left + pad_x:right - pad_x]
            if center.size == 0:
                continue
            gray = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
            measurements.append(float(gray.std()))
            crops.append(center)
            geometry.append((left, top, right, bottom))
    if len(geometry) < 4 or len(geometry) > 80:
        return [], {"rows": 0, "vehicles": 0, "inferred_empty": 0, "reliability": "Insufficient evidence", "engine": "painted-grid"}

    low, high = float(min(measurements)), float(max(measurements))
    # Empty asphalt bays are nearly uniform; even dark cars add edges and
    # grayscale variation. Keep the threshold near the empty cluster rather
    # than halfway toward bright white vehicles.
    threshold = max(12.0, low + 0.12 * (high - low))
    learned_probabilities = None
    if occupancy_model is not None:
        from parking_classifier import batch_features
        learned_probabilities = occupancy_model.predict_proba(batch_features(crops))[:, 1]
    results: list[SlotPrediction] = []
    occupied_count = 0
    geometry_index = 0
    for row, column_count in enumerate(row_columns, 1):
        for column in range(1, column_count + 1):
            (left, top, right, bottom), variation = geometry[geometry_index], measurements[geometry_index]
            geometry_index += 1
            probability = float(learned_probabilities[geometry_index - 1]) if learned_probabilities is not None else None
            occupied = probability >= .5 if probability is not None else variation >= threshold
            occupied_count += int(occupied)
            if probability is not None:
                confidence = probability if occupied else 1.0 - probability
            else:
                distance = min(0.27, abs(variation - threshold) / 45.0)
                confidence = 0.70 + distance
            results.append(SlotPrediction(
                slot_id=f"R{row:02d}-S{column:02d}", status="Occupied" if occupied else "Available",
                confidence=confidence, occupied_probability=confidence if occupied else 1.0 - confidence,
                points=[[left / width, top / height], [right / width, top / height],
                        [right / width, bottom / height], [left / width, bottom / height]],
            ))
    return results, {
        "rows": len(y_pairs), "vehicles": occupied_count,
        "inferred_empty": len(results) - occupied_count,
        "reliability": "Strong" if len(results) >= 6 else "Limited",
        "engine": "PKLot trained classifier" if occupancy_model is not None else "painted-grid heuristic",
    }


def load_yolo(weights: str | Path = "models/yolo11n-obb.onnx") -> Any:
    """Load the bundled ONNX detector without PyTorch or a web download."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime is not installed. Run: pip install -r requirements.txt"
        ) from exc
    path = Path(weights)
    if not path.exists():
        raise RuntimeError(f"Bundled YOLO model is missing: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def detect_vehicles(image: np.ndarray, model: Any, confidence: float = 0.045) -> list[VehicleDetection]:
    import cv2

    input_shape = model.get_inputs()[0].shape
    size = int(input_shape[-1]) if isinstance(input_shape[-1], int) else 640
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    blob = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))[None]
    input_name = model.get_inputs()[0].name
    raw = np.asarray(model.run(None, {input_name: blob})[0])
    rows = raw[0].T if raw.ndim == 3 and raw.shape[1] < raw.shape[2] else raw.reshape(-1, raw.shape[-1])
    is_aerial_obb = rows.shape[1] == 20
    is_custom_obb = rows.shape[1] == 7
    vehicle_classes = ({1} if is_custom_obb else DOTA_VEHICLE_CLASSES if is_aerial_obb else COCO_VEHICLE_CLASSES)
    boxes: list[list[int]] = []
    scores: list[float] = []
    classes: list[int] = []
    for row in rows:
        class_scores = row[4:-1] if is_aerial_obb or is_custom_obb else row[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if class_id not in vehicle_classes or score < confidence:
            continue
        cx, cy, bw, bh = map(float, row[:4])
        boxes.append([round(cx - bw / 2), round(cy - bh / 2), round(bw), round(bh)])
        scores.append(score)
        classes.append(class_id)
    selected = cv2.dnn.NMSBoxes(boxes, scores, confidence, 0.55)
    class_names = ({1: "vehicle"} if is_custom_obb else {9: "large vehicle", 10: "small vehicle"} if is_aerial_obb
                   else {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
    output: list[VehicleDetection] = []
    for index in np.asarray(selected).reshape(-1):
        x, y, bw, bh = boxes[int(index)]
        output.append(VehicleDetection(
            box=(max(0.0, x / size), max(0.0, y / size), min(1.0, (x + bw) / size), min(1.0, (y + bh) / size)),
            confidence=scores[int(index)], class_name=class_names[classes[int(index)]],
        ))
    return output


def _cluster_rows(detections: list[VehicleDetection]) -> list[list[VehicleDetection]]:
    """Group vehicles that share approximately the same image-space row."""
    ordered = sorted(detections, key=lambda d: (d.box[1] + d.box[3]) / 2)
    rows: list[list[VehicleDetection]] = []
    for detection in ordered:
        cy = (detection.box[1] + detection.box[3]) / 2
        height = detection.box[3] - detection.box[1]
        best: list[VehicleDetection] | None = None
        best_distance = float("inf")
        for row in rows:
            row_cy = float(np.median([(d.box[1] + d.box[3]) / 2 for d in row]))
            row_h = float(np.median([d.box[3] - d.box[1] for d in row]))
            distance = abs(cy - row_cy)
            if distance < max(0.055, 0.48 * max(height, row_h)) and distance < best_distance:
                best, best_distance = row, distance
        (best if best is not None else rows.append([]) or rows[-1]).append(detection)
    return rows


def infer_parking_spaces(detections: list[VehicleDetection]) -> tuple[list[SlotPrediction], dict[str, Any]]:
    """Create occupied bays and infer empty bays in visible gaps.

    This deliberately avoids inventing spaces beyond the outermost detected
    vehicles. A gap is called empty only when it is large enough for at least
    one typical vehicle in that row.
    """
    if not detections:
        return [], {"rows": 0, "vehicles": 0, "inferred_empty": 0, "reliability": "Insufficient evidence"}
    predictions: list[SlotPrediction] = []
    inferred_empty = 0
    rows = _cluster_rows(detections)
    for row_index, row in enumerate(rows, 1):
        row = sorted(row, key=lambda d: (d.box[0] + d.box[2]) / 2)
        widths = [d.box[2] - d.box[0] for d in row]
        heights = [d.box[3] - d.box[1] for d in row]
        slot_w = min(0.30, max(0.055, float(np.median(widths)) * 1.28))
        slot_h = min(0.55, max(0.10, float(np.median(heights)) * 1.12))
        centers = [(d.box[0] + d.box[2]) / 2 for d in row]
        row_cy = float(np.median([(d.box[1] + d.box[3]) / 2 for d in row]))
        slots: list[tuple[float, VehicleDetection | None]] = [(center, detection) for center, detection in zip(centers, row, strict=True)]
        for left, right in zip(centers, centers[1:]):
            missing = max(0, round((right - left) / slot_w) - 1)
            for number in range(missing):
                slots.append((left + (number + 1) * (right - left) / (missing + 1), None))
                inferred_empty += 1
        for slot_index, (cx, detection) in enumerate(sorted(slots, key=lambda item: item[0]), 1):
            x1, x2 = max(0.0, cx - slot_w / 2), min(1.0, cx + slot_w / 2)
            y1, y2 = max(0.0, row_cy - slot_h / 2), min(1.0, row_cy + slot_h / 2)
            occupied = detection is not None
            confidence = detection.confidence if detection else 0.72
            predictions.append(SlotPrediction(
                slot_id=f"R{row_index:02d}-S{slot_index:02d}",
                status="Occupied" if occupied else "Available",
                confidence=float(confidence),
                occupied_probability=float(confidence if occupied else 1.0 - confidence),
                points=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            ))
    reliability = "Strong" if len(detections) >= 4 else "Limited"
    return predictions, {"rows": len(rows), "vehicles": len(detections), "inferred_empty": inferred_empty, "reliability": reliability}


def analyse_automatic(image: np.ndarray, model: Any | None, confidence: float = 0.045, occupancy_model: Any | None = None):
    marked_slots, marked_info = infer_marked_parking_spaces(image, occupancy_model)
    if marked_slots:
        return marked_slots, marked_info
    if model is None:
        return [], {
            "rows": 0, "vehicles": 0, "inferred_empty": 0,
            "reliability": "Detector unavailable", "engine": "painted-grid",
        }
    detections = detect_vehicles(image, model, confidence)
    slots, diagnostics = infer_parking_spaces(detections)
    diagnostics["engine"] = "aerial-yolo"
    return slots, diagnostics
