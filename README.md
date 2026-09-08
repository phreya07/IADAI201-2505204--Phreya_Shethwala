# ParkVision AI

ParkVision AI is a Streamlit smart-parking application that detects visible
parking spaces in a complete aerial or elevated photograph and classifies each
space as `Available` or `Occupied`. The application uses a custom two-class
YOLO11 full-scene detector; it does not require rows, columns, a saved camera
layout, or manual calibration.

# Intelligent Urban Parking Analytics and Space Optimisation Platform

ParkVision AI is a computer-vision application designed to analyse parking-lot photographs and convert them into useful parking information. Instead of manually counting vehicles and empty spaces, a user can upload an aerial or elevated image and allow the system to detect visible parking spaces, classify them as occupied or available, and present the results through a professional dashboard.

The project goes beyond basic object detection. It combines parking-space analysis with operational alerts, demand forecasting, revenue scenarios, sustainability estimates, quality review and downloadable reports. The aim is to help parking operators, schools, businesses, shopping centres and smart-city teams make faster and more informed parking decisions.

# Problem Being Addressed

Drivers often spend unnecessary time searching for parking because they do not know where spaces are available. This increases congestion, fuel consumption, waiting time and emissions. Parking managers may also lack a simple way to monitor capacity or recognise when overflow arrangements are required.

Many parking systems depend on physical sensors installed in every bay. Although sensors can be reliable, installing and maintaining them may be expensive. ParkVision AI explores a camera-based alternative by using computer vision to estimate parking availability from an image.

# Main capabilities

Automatic full-scene empty/occupied space detection

Annotated availability map and slot-level confidence

Total, available, occupied and utilisation summaries

Congestion alerts and operational recommendations

Two-hour demand scenarios

Revenue and sustainability planning estimates

CSV, PNG and human-review feedback downloads

Verified full-scene model evaluation panel

# Limitations

Best results require a clear aerial or elevated parking-lot photograph.

Very small, blurred, dark or heavily obstructed spaces may be missed.

Parking styles outside the PKLot training domain can reduce performance.

Operational, financial and sustainability outputs are planning estimates.

Low-confidence predictions should be reviewed before operational use.

# Future Improvements

Train with additional parking datasets and camera angles.

Add more labelled examples from the intended deployment locations.

Support live CCTV and video analysis.

Track vehicles and availability changes over time.

Store historical utilisation information in a database.

Add user authentication and administrator accounts.

Connect the system to digital entrance signs or a mobile application.

Use human corrections for controlled model retraining.

# Conclusion

ParkVision AI demonstrates how computer vision can turn a parking-lot image into practical information. It combines automated availability detection with quality review, forecasting, operational guidance and reporting. The project also separates verified test performance, prediction confidence and image-specific agreement so that its results are presented transparently. With broader labelled training data and location-specific fine-tuning, the system could develop into a useful component of a smart parking management platform.






streamlit - https://iadai201-2505204--phreyashethwala-ulexplxinsemmybur7ymtr.streamlit.app/
