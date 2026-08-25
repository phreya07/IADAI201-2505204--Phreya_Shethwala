# ParkVision AI — Automatic Parking Analytics

ParkVision is a Streamlit smart-city application that accepts a parking-lot
photograph, detects painted bays automatically and uses aerial YOLO as a
fallback, annotates the image and reports capacity,
utilisation, congestion, forecasts and urban-impact estimates.

## What changed

The primary app no longer uses a fixed 12-slot grid. Users do not enter rows,
columns, margins or layout JSON. The flow is:

```text
Original parking photograph
        ↓
Painted-bay grid detection; aerial YOLO fallback
        ↓
Automatic bay classification or row/gap inference
        ↓
Occupied/available estimates, map, reports and insights
```

The earlier PKLot MobileNetV2 model and its genuine evaluation evidence remain
in the repository for the classification part of the assessment. The automatic
app uses YOLO because a crop classifier cannot locate spaces in a new image.
It loads `models/parking_best.onnx` when a custom trained detector is present;
otherwise it uses the included aerial DOTA vehicle detector.

## Windows — easiest method

Extract the complete ZIP and double-click `RUN_APP.bat`. It creates `.venv`,
installs the correct packages and opens Streamlit. The official `yolo11n-obb.onnx`
model is already included, so analysis does not download weights at runtime.

Manual PowerShell commands:

```powershell
cd "C:\path\to\ParkVision_AI"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the `Local URL` printed by Streamlit, normally `http://localhost:8501`.
The message `Uvicorn server started` alone is not an error; keep that terminal
open while using the browser.

## How to obtain the best result

1. Upload the original JPG or PNG—not a screenshot containing old boxes,
   Streamlit controls, captions or labels.
2. Use a clear image showing most of the parking area.
3. Keep the default aerial confidence at `0.045`. Lower it slightly if a real vehicle
   is missed; raise it if background objects are detected incorrectly.
4. Check the annotated map and use the Review tab to correct uncertain results.

The program deliberately does not invent outer empty spaces that have no visual
evidence. Empty gaps are inferred between vehicles in a discovered row. A
completely empty parking lot needs a parking-line/space segmentation model
trained on full-scene polygon annotations; vehicle-only YOLO cannot know the
capacity of an unmarked or fully empty area.

## Secure Kaggle setup

The Kaggle API key downloads training data; it does not improve predictions by
itself. Never hard-code or commit it. For Streamlit Cloud, add:

```toml
KAGGLE_USERNAME = "your_username"
KAGGLE_KEY = "your_private_key"
```

under **App settings → Secrets**. For local retraining:

```powershell
python setup_kaggle.py "C:\path\to\kaggle.json"
```

## Included features

- automatic multi-layout vehicle detection;
- inferred parking rows and visible empty gaps;
- total, occupied, available and utilisation metrics;
- congestion recommendations;
- annotated PNG and CSV downloads;
- 30-minute demand forecast;
- time, fuel and CO₂ planning estimates;
- confidence reporting and human-review feedback export;
- safe error messages for invalid images, missing packages and no detections.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

## Custom full-scene YOLO training

For the strongest generalisation, annotate full parking scenes in YOLO OBB
format with `parking_space` and `vehicle` polygons. Keep source cameras grouped
when making the 70/15/15 train/validation/test split. Place the images and labels
under `data/parking_yolo` as described in
`configuration/parking_dataset.yaml`, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe train_yolo_detector.py
```

Review test precision, recall and mAP before copying the exported model to
`models/parking_best.onnx`. The app automatically prefers that file.

## Verified classification evidence

The included PKLot MobileNetV2 experiment achieved 93.52% accuracy, 93.86% F1
and 99.07% occupied recall on its untouched 1,080-image test set. These values
apply only to that crop-classification test set; they are not advertised as the
accuracy of arbitrary full-scene uploads.

## Honest limitations

No vision system can guarantee 100% accuracy for every internet photograph.
Heavy occlusion, night scenes, extreme camera angles, tiny vehicles and unusual
parking geometry may reduce performance. For production-grade detection of
every empty bay—including completely empty lots—the next training stage should
use full parking images with polygon annotations for both `parking_space` and
`vehicle`, evaluated using precision, recall and mAP on an unseen test set.
