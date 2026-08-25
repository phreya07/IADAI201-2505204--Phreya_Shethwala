"""Train, calibrate, evaluate, and export the ParkVision MobileNetV2 model."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
MODEL_PATH = MODELS / "parkvision_model.keras"
PHASE1_PATH = MODELS / "parkvision_phase1.keras"
METADATA_PATH = MODELS / "model_metadata.json"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 32
tf.keras.utils.set_random_seed(SEED)


def load_dataset(split: str, shuffle: bool):
    path = DATA / split
    if not path.exists():
        raise SystemExit(f"Missing dataset folder: {path}. Run prepare_dataset.py first.")
    return tf.keras.utils.image_dataset_from_directory(
        path, image_size=(224, 224), batch_size=BATCH_SIZE,
        label_mode="binary", shuffle=shuffle, seed=SEED,
    )


train = load_dataset("train", True)
validation = load_dataset("validation", False)
test = load_dataset("test", False)
if train.class_names != ["empty", "occupied"]:
    raise SystemExit(f"Expected class order ['empty', 'occupied']; found {train.class_names}.")

train = train.prefetch(tf.data.AUTOTUNE)
validation = validation.prefetch(tf.data.AUTOTUNE)
test = test.prefetch(tf.data.AUTOTUNE)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.12),
    tf.keras.layers.RandomTranslation(0.06, 0.06),
    tf.keras.layers.RandomContrast(0.18),
    tf.keras.layers.RandomBrightness(0.12, value_range=(0, 255)),
], name="training_augmentation")

base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base.trainable = False
inputs = tf.keras.Input((224, 224, 3), name="parking_slot_rgb")
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.35)(x)
x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="occupied_probability")(x)
model = tf.keras.Model(inputs, outputs, name="ParkVision_MobileNetV2")


def compile_model(learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )


def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.35, patience=2, min_lr=1e-7),
    ]


compile_model(8e-4)
history1 = model.fit(train, validation_data=validation, epochs=25, callbacks=callbacks())
phase1_loss = min(history1.history["val_loss"])
model.save(PHASE1_PATH)

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False
for layer in base.layers[-40:]:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
compile_model(8e-6)
history2 = model.fit(train, validation_data=validation, epochs=15, callbacks=callbacks())
phase2_loss = min(history2.history["val_loss"])

if phase2_loss > phase1_loss:
    model = tf.keras.models.load_model(PHASE1_PATH, compile=False)
    selected_phase = "frozen-base"
else:
    selected_phase = "fine-tuned"
model.save(MODEL_PATH)
PHASE1_PATH.unlink(missing_ok=True)


def labels(dataset) -> np.ndarray:
    return np.concatenate([batch.numpy().reshape(-1) for _, batch in dataset]).astype(int)


validation_true = labels(validation)
validation_probability = np.asarray(model.predict(validation, verbose=1)).reshape(-1)
thresholds = np.linspace(0.20, 0.80, 121)
threshold_scores = [(float(t), f1_score(validation_true, validation_probability >= t, average="macro")) for t in thresholds]
best_threshold, validation_f1 = max(threshold_scores, key=lambda item: (item[1], -abs(item[0] - 0.5)))

test_true = labels(test)
test_probability = np.asarray(model.predict(test, verbose=1)).reshape(-1)
test_prediction = (test_probability >= best_threshold).astype(int)
metrics = {
    "accuracy": float(accuracy_score(test_true, test_prediction)),
    "precision": float(precision_score(test_true, test_prediction, zero_division=0)),
    "recall": float(recall_score(test_true, test_prediction, zero_division=0)),
    "f1_score": float(f1_score(test_true, test_prediction, zero_division=0)),
}

report = classification_report(test_true, test_prediction, target_names=["empty", "occupied"], output_dict=True, zero_division=0)
(RESULTS / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(RESULTS / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

metadata = {
    "model": "MobileNetV2 transfer learning",
    "input_size": [224, 224, 3],
    "class_names": ["empty", "occupied"],
    "occupied_class_index": 1,
    "occupied_threshold": round(float(best_threshold), 4),
    "validation_macro_f1": round(float(validation_f1), 4),
    "selected_training_phase": selected_phase,
    "test_metrics": {key: round(value, 4) for key, value in metrics.items()},
}
METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

history = {key: history1.history.get(key, []) + history2.history.get(key, []) for key in set(history1.history) | set(history2.history)}
figure, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.get("accuracy", []), label="Train")
axes[0].plot(history.get("val_accuracy", []), label="Validation")
axes[0].set(title="Model accuracy", xlabel="Epoch", ylabel="Accuracy")
axes[0].legend()
axes[1].plot(history.get("loss", []), label="Train")
axes[1].plot(history.get("val_loss", []), label="Validation")
axes[1].set(title="Model loss", xlabel="Epoch", ylabel="Loss")
axes[1].legend()
figure.tight_layout()
figure.savefig(RESULTS / "training_history.png", dpi=180)
plt.close(figure)

matrix = confusion_matrix(test_true, test_prediction)
figure, axis = plt.subplots(figsize=(5.5, 4.6))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Empty", "Occupied"], yticklabels=["Empty", "Occupied"], ax=axis)
axis.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix · threshold {best_threshold:.2f}")
figure.tight_layout()
figure.savefig(RESULTS / "confusion_matrix.png", dpi=180)
plt.close(figure)

print(json.dumps(metadata, indent=2))
print(f"Saved calibrated model to: {MODEL_PATH}")
