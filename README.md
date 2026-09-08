# ParkVision AI

ParkVision AI is a Streamlit smart-parking application that detects visible
parking spaces in a complete aerial or elevated photograph and classifies each
space as `Available` or `Occupied`. The application uses a custom two-class
YOLO11 full-scene detector; it does not require rows, columns, a saved camera
layout, or manual calibration.

## Main capabilities

- Automatic full-scene empty/occupied space detection
- Annotated availability map and slot-level confidence
- Total, available, occupied and utilisation summaries
- Congestion alerts and operational recommendations
- Two-hour demand scenarios
- Revenue and sustainability planning estimates
- CSV, PNG and human-review feedback downloads
- Verified full-scene model evaluation panel

## Model and dataset

The detector was trained from PKLot COCO annotations converted to YOLO format.

| Split | Complete scenes | Labelled spaces |
|---|---:|---:|
| Training | 1,600 | 93,352 |
| Validation | 400 | 24,712 |
| Test | 400 | 23,248 |

The untouched test split produced:

| Metric | Result |
|---|---:|
| Precision | 89.51% |
| Recall | 92.25% |
| mAP50 | 97.01% |
| mAP50-95 | 65.17% |
| Empty-space mAP50 | 96.93% |
| Occupied-space mAP50 | 97.08% |

mAP measures full-scene localisation and classification. It is not the same as
an individual box's confidence or universal accuracy on every photograph.

## Project structure

```text
ParkVision_AI/
├── .streamlit/config.toml
├── app.py
├── automatic_parking.py
├── core.py
├── convert_coco_fullscene.py
├── train_yolo_detector.py
├── configuration/parking_dataset.yaml
├── models/
│   ├── parking_best.onnx
│   └── fullscene_metrics.json
├── results/
├── tests/
├── requirements.txt
├── requirements-training.txt
└── runtime.txt
```

The raw dataset and private Kaggle credential are intentionally excluded.

## Local installation

Use Python 3.11, create a virtual environment and install the runtime packages.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment first with:

```powershell
.venv\Scripts\Activate.ps1
```

## Streamlit deployment

1. Upload the complete project to GitHub.
2. Confirm that `models/parking_best.onnx` is present.
3. Create or reboot the Streamlit Community Cloud application.
4. Set the main file to `app.py`.

No Kaggle credential is required during prediction or deployment.

### If Streamlit says "Error installing requirements"

1. Replace the repository contents with this package. Do not keep an old
   `packages.txt` or a second `requirements.txt` in another folder.
2. Open **Manage app**, select **Reboot app**, and wait for installation.
3. Keep `runtime.txt` at the repository root so Python 3.12 is selected.

## Reproducing training

```bash
python -m pip install -r requirements-training.txt
python convert_coco_fullscene.py /path/to/pklot --output data/parking_fullscene
python train_yolo_detector.py
```

Update `configuration/parking_dataset.yaml` if the processed dataset is stored
elsewhere. The training script evaluates the best checkpoint and exports the
deployable ONNX model.

## Limitations

- Best results require a clear aerial or elevated parking-lot photograph.
- Very small, blurred, dark or heavily obstructed spaces may be missed.
- Parking styles outside the PKLot training domain can reduce performance.
- Operational, financial and sustainability outputs are planning estimates.
- Low-confidence predictions should be reviewed before operational use.

## Security

Never upload `kaggle.json`, secrets, virtual environments, raw datasets or
Python cache files to GitHub.
