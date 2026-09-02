# ParkVision AI

ParkVision AI is a computer-vision parking analytics platform developed for smart urban mobility. It analyses a parking-lot image from a supported fixed camera, classifies each configured parking space as available or occupied, and converts the results into clear utilisation metrics and congestion guidance.

The system combines a MobileNetV2 image-classification model with a professional Streamlit dashboard. It is designed to help drivers, parking operators, and urban planners understand parking availability quickly and make better mobility decisions.

# Key Features

Slot-level classification of available and occupied parking spaces

MobileNetV2 transfer learning with two-stage training and fine-tuning

Automatic validation-based calibration of the occupancy threshold

Annotated parking map with colour-coded space boundaries

Real-time availability, occupancy, utilisation, and congestion metrics

Confidence scores and low-confidence prediction warnings

Operational recommendations based on parking demand

Downloadable annotated image and slot-level CSV report

Human-review interface for correcting predictions and exporting feedback

Input validation, model validation, and safe error handling

Responsive Streamlit interface for local or cloud deployment

# System Workflow

The user uploads a JPG or PNG image from the configured parking camera.

The application reads normalized parking-space polygons from a layout JSON file.

Each parking space is cropped and resized to 224 × 224 pixels.

MobileNetV2 calculates the probability that each space is occupied.

A calibrated threshold converts each probability into an Available or Occupied result.

The system annotates the image and calculates availability, utilisation, confidence, and congestion statistics.

Results can be reviewed and downloaded for reporting or future model improvement.

# Limitations

The current version requires a fixed camera and a matching polygon layout.

It does not automatically detect parking-space boundaries in an arbitrary image.

Major camera movement can make the configured polygons inaccurate.

Strong shadows, severe glare, night scenes, rain, and vehicle occlusion may reduce accuracy.

Performance depends on the diversity and quality of the training data.

Predictions with low confidence should be reviewed manually.

# Future Improvements

Automatic parking-space detection for unseen camera views

Live video and CCTV-stream processing

Temporal smoothing across consecutive video frames

Expanded training data for night, rain, and extreme lighting conditions

Multi-camera parking management

Historical utilisation analytics and demand forecasting

Database integration and operator alerts

Continuous retraining using verified human-review feedback




streamlit - https://iadai201-2505204--phreyashethwala-ulexplxinsemmybur7ymtr.streamlit.app/
