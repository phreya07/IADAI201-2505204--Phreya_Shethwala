"""Small, deployment-friendly feature extractor for parking-space crops."""
from __future__ import annotations

import numpy as np
from PIL import Image


def crop_features(crop: np.ndarray) -> np.ndarray:
    image = np.asarray(Image.fromarray(crop).resize((48, 48), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gray = image.mean(axis=2)
    values: list[float] = []
    # Global and spatial colour/texture summaries.
    for rows in (1, 2, 4):
        step = 48 // rows
        for y in range(rows):
            for x in range(rows):
                patch = image[y*step:(y+1)*step, x*step:(x+1)*step]
                values.extend(patch.mean(axis=(0, 1)).tolist())
                values.extend(patch.std(axis=(0, 1)).tolist())
    for channel in range(3):
        hist, _ = np.histogram(image[..., channel], bins=12, range=(0, 1), density=True)
        values.extend((hist / max(hist.sum(), 1e-8)).tolist())
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    magnitude = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + np.pi) % np.pi
    for y in range(0, 48, 12):
        for x in range(0, 48, 12):
            hist, _ = np.histogram(angle[y:y+12, x:x+12], bins=9, range=(0, np.pi), weights=magnitude[y:y+12, x:x+12])
            values.extend((hist / max(hist.sum(), 1e-8)).tolist())
    values.extend([float(gray.mean()), float(gray.std()), float(magnitude.mean()), float((magnitude > .12).mean())])
    return np.asarray(values, dtype=np.float32)


def batch_features(crops: list[np.ndarray]) -> np.ndarray:
    return np.stack([crop_features(crop) for crop in crops])
