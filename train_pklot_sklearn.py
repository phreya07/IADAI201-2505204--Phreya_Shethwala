"""Train/evaluate the lightweight PKLot classifier from COCO annotations."""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from parking_classifier import crop_features

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=Path)
parser.add_argument("--train-per-class", type=int, default=12000)
parser.add_argument("--validation-per-class", type=int, default=2500)
parser.add_argument("--test-per-class", type=int, default=2500)
parser.add_argument("--models", type=Path, default=Path(__file__).parent / "models")
args = parser.parse_args()
random.seed(42)


def load_split(name: str, per_class: int):
    folder = args.dataset / ("valid" if name == "validation" else name)
    coco = json.loads((folder / "_annotations.coco.json").read_text())
    images = {item["id"]: item["file_name"] for item in coco["images"]}
    by_class = {1: [], 2: []}
    for item in coco["annotations"]:
        if item["category_id"] in by_class:
            by_class[item["category_id"]].append(item)
    selected = []
    for category in (1, 2):
        random.shuffle(by_class[category])
        selected.extend(by_class[category][:per_class])
    random.shuffle(selected)
    features, labels, cached_id, cached_image = [], [], None, None
    for index, item in enumerate(selected, 1):
        if item["image_id"] != cached_id:
            cached_id = item["image_id"]
            cached_image = np.asarray(Image.open(folder / images[cached_id]).convert("RGB"))
        x, y, w, h = map(int, item["bbox"])
        crop = cached_image[max(0,y):max(0,y)+max(2,h), max(0,x):max(0,x)+max(2,w)]
        if crop.size:
            features.append(crop_features(crop)); labels.append(item["category_id"] - 1)
        if index % 5000 == 0: print(name, index, "/", len(selected), flush=True)
    return np.stack(features), np.asarray(labels)


x_train, y_train = load_split("train", args.train_per_class)
x_validation, y_validation = load_split("validation", args.validation_per_class)
x_test, y_test = load_split("test", args.test_per_class)
model = HistGradientBoostingClassifier(max_iter=180, learning_rate=.09, max_leaf_nodes=31, l2_regularization=.25, random_state=42)
model.fit(x_train, y_train)

def metrics(x, y):
    prediction = model.predict(x)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(y, prediction)), "precision": float(precision), "recall": float(recall), "f1_score": float(f1), "samples": int(len(y))}

args.models.mkdir(parents=True, exist_ok=True)
joblib.dump(model, args.models / "occupancy_classifier.joblib", compress=3)
metadata = {"model":"Histogram Gradient Boosting on PKLot crop features", "dataset":"PKLot", "classes":["empty","occupied"], "validation_metrics":metrics(x_validation,y_validation), "test_metrics":metrics(x_test,y_test), "test_split":"Official Kaggle/Roboflow test folder", "seed":42}
(args.models / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
