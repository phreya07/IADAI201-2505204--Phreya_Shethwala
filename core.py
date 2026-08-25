"""Reusable inference and parking-insight logic for ParkVision AI."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, UnidentifiedImageError


@dataclass(frozen=True)
class SlotPrediction:
    slot_id: str
    status: str
    confidence: float
    occupied_probability: float
    points: list[list[float]]


def read_image(data: bytes) -> np.ndarray:
    if not data or len(data) > 25 * 1024 * 1024:
        raise ValueError("Upload a non-empty image smaller than 25 MB.")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.verify() if False else None
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid JPG or PNG image.") from exc
    arr = np.asarray(image)
    if min(arr.shape[:2]) < 100:
        raise ValueError("Image is too small. Use an image at least 100 × 100 pixels.")
    return arr


def load_layout(source: str | Path | bytes) -> dict[str, Any]:
    raw = source if isinstance(source, bytes) else Path(source).read_bytes()
    try:
        layout = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Layout must be valid JSON.") from exc
    slots = layout.get("slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("Layout JSON must contain a non-empty 'slots' list.")
    ids: set[str] = set()
    for slot in slots:
        sid, points = str(slot.get("id", "")).strip(), slot.get("points")
        if not sid or sid in ids:
            raise ValueError("Every slot needs a unique, non-empty id.")
        ids.add(sid)
        if not isinstance(points, list) or len(points) < 4:
            raise ValueError(f"Slot {sid} needs at least four polygon points.")
        for point in points:
            if len(point) != 2 or not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in point):
                raise ValueError(f"Slot {sid} points must be normalized [x, y] values from 0 to 1.")
    return layout


def crop_polygon(image: np.ndarray, points: list[list[float]]) -> np.ndarray:
    h, w = image.shape[:2]
    polygon = np.array([[round(x * w), round(y * h)] for x, y in points], dtype=np.int32)
    x, y = np.maximum(polygon.min(axis=0), 0)
    x2, y2 = np.minimum(polygon.max(axis=0) + 1, [w, h])
    crop = image[y:y2, x:x2]
    if crop.size == 0:
        raise ValueError("A configured slot falls outside the image.")
    return crop


def prepare_batch(crops: list[np.ndarray]) -> np.ndarray:
    """Resize crops for the saved model.

    Keep values in the 0-255 range because MobileNetV2 preprocessing is part
    of the saved Keras model. Applying it here too would normalize twice and
    produce unreliable predictions.
    """
    batch = np.stack([
        np.asarray(Image.fromarray(c).resize((224, 224), Image.Resampling.LANCZOS))
        for c in crops
    ]).astype("float32")
    return batch


def demo_probabilities(crops: list[np.ndarray]) -> np.ndarray:
    """Deterministic visual heuristic used only when a trained model is absent."""
    values = []
    for crop in crops:
        pil = Image.fromarray(crop)
        gray = np.asarray(pil.convert("L"))
        edges = np.asarray(pil.convert("L").filter(ImageFilter.FIND_EDGES))
        edge_density = float(np.mean(edges > 70))
        contrast = float(gray.std() / 128.0)
        saturation = float(np.asarray(pil.convert("HSV"))[..., 1].mean() / 255.0)
        score = 0.18 + 2.0 * edge_density + 0.38 * contrast + 0.20 * saturation
        values.append(float(np.clip(score, 0.05, 0.95)))
    return np.asarray(values)


def topdown_visual_guard(crops: list[np.ndarray], config: dict[str, Any]) -> np.ndarray:
    """Detect a centred vehicle in clean, perpendicular parking bays.

    This guard is layout-specific and is intentionally disabled unless the
    layout opts in. It complements the PKLot classifier when a clean overhead
    camera differs strongly from PKLot's angled camera domain.
    """
    threshold = float(config.get("center_grayscale_std_threshold", 15.0))
    probabilities = []
    for crop in crops:
        h, w = crop.shape[:2]
        center = crop[round(0.15 * h):round(0.85 * h), round(0.25 * w):round(0.75 * w)]
        gray = np.asarray(Image.fromarray(center).convert("L"), dtype=np.float32)
        variation = float(gray.std())
        probability = 0.05 if variation < threshold else min(0.99, 0.75 + (variation - threshold) / 40.0)
        probabilities.append(probability)
    return np.asarray(probabilities, dtype=np.float32)


def predict_slots(image: np.ndarray, layout: dict[str, Any], model: Any | None, threshold: float) -> list[SlotPrediction]:
    crops = [crop_polygon(image, slot["points"]) for slot in layout["slots"]]
    guard = layout.get("topdown_visual_guard", {})
    if model is None:
        probabilities = demo_probabilities(crops)
        if guard.get("enabled"):
            probabilities = np.maximum(probabilities, topdown_visual_guard(crops, guard))
    else:
        probabilities = np.asarray(model.predict(prepare_batch(crops), verbose=0)).reshape(-1)
        if len(probabilities) != len(crops) or not np.all(np.isfinite(probabilities)):
            raise ValueError("The model returned invalid prediction values.")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        if guard.get("enabled"):
            probabilities = np.maximum(probabilities, topdown_visual_guard(crops, guard))
    results = []
    for slot, probability in zip(layout["slots"], probabilities, strict=True):
        occupied = float(probability) >= threshold
        results.append(SlotPrediction(
            slot_id=slot["id"], status="Occupied" if occupied else "Available",
            confidence=float(probability if occupied else 1 - probability),
            occupied_probability=float(probability), points=slot["points"],
        ))
    return results


def summarize(results: list[SlotPrediction]) -> dict[str, Any]:
    total = len(results)
    occupied = sum(r.status == "Occupied" for r in results)
    available = total - occupied
    rate = 100 * occupied / total if total else 0.0
    if rate < 40:
        level, message = "Low", "Good availability — drivers may proceed."
    elif rate <= 75:
        level, message = "Moderate", "Parking is filling up — proceed soon."
    else:
        level, message = "High", "Very limited availability — consider another parking area."
    if available == 0:
        level, message = "Full", "Parking is full — please try another area."
    return {"total": total, "occupied": occupied, "available": available, "occupancy": rate, "level": level, "message": message}


def annotate(image: np.ndarray, results: list[SlotPrediction]) -> np.ndarray:
    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    scale = max(0.55, min(w, h) / 850)
    font = ImageFont.load_default(size=max(11, round(14 * scale)))
    for result in results:
        pts = [(round(x * w), round(y * h)) for x, y in result.points]
        color = (42, 190, 137) if result.status == "Available" else (239, 83, 100)
        draw.polygon(pts, fill=(*color, 42), outline=(*color, 255), width=max(2, round(3 * scale)))
        x, y = min(pts, key=lambda point: point[1])
        label = f"{result.slot_id}  {result.status}  {result.confidence:.0%}"
        box = draw.textbbox((x, max(2, y - 22)), label, font=font)
        draw.rounded_rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2), 4, fill=(11, 23, 54, 220))
        draw.text((x, max(2, y - 22)), label, fill="white", font=font)
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))
