# Intelligent Urban Parking Analytics and Space Optimisation Platform

ParkVision AI is a smart parking management application that uses artificial intelligence and computer vision to identify available and occupied parking spaces from an uploaded parking-lot image. It converts visual information into a clear parking summary so that drivers and parking operators can understand the current availability of spaces quickly.

The project was developed to address a common urban problem: drivers often spend unnecessary time searching for an empty parking space. This increases traffic congestion, travel time, fuel consumption, and air pollution. ParkVision AI provides a simple digital solution by analysing individual parking spaces and presenting the information through a professional Streamlit dashboard.

# Purpose of the Project

The main purpose of ParkVision AI is to make parking management faster, clearer, and more efficient. Instead of checking every parking space manually, the system analyses an image and displays:

The total number of configured parking spaces

The number of available spaces

The number of occupied spaces

The overall parking utilisation percentage

The current congestion level

A recommendation based on the parking situation

The application is useful as a smart-city prototype because it shows how artificial intelligence can support better traffic management and improve the experience of drivers.

# Problem Statement

In busy urban areas, drivers may enter a parking facility without knowing whether a space is available. They often need to drive around the area several times before finding an empty slot. This creates avoidable congestion and wastes time and fuel.

Traditional parking management may require manual checking or expensive sensors installed in every parking space. ParkVision AI explores an image-based approach in which a camera view and an artificial-intelligence model are used to classify individual parking spaces.

# Proposed Solution

ParkVision AI accepts a parking-lot image taken from a supported fixed-camera position. The image is divided into predefined parking areas using a layout file. Each parking-space image is then processed by a trained MobileNetV2 model and classified as either:

Available

Occupied

The application displays the predictions on an annotated parking map. Available spaces are shown in green, while occupied spaces are shown in red. It also calculates the utilisation of the parking area and provides simple operational guidance.

# Main Features

# Parking-Space Detection

The application analyses every configured parking slot separately. This provides a more useful result than giving only the total number of vehicles in the image.

# Occupancy Classification

Each parking space is classified as available or occupied using a MobileNetV2 deep-learning model.

# Annotated Parking Map

The system draws coloured boundaries and labels over the original image. This allows users to understand the result visually without reading every individual prediction.

# Parking Summary

The dashboard automatically displays the total, available, and occupied spaces together with the percentage of the parking area currently in use.

# Congestion Analysis

The application interprets the utilisation percentage and classifies the parking condition as Low, Moderate, High, or Full.

# Smart Recommendations

ParkVision AI provides guidance based on the current parking condition. For example, it can inform users that capacity is healthy, warn that the area is filling up, or recommend redirecting incoming drivers when the parking area is full.

# Confidence Information

The system displays confidence information for its predictions. Low-confidence results are identified so that a human user can review uncertain parking spaces.

# Human Review

Users can correct an inaccurate parking-space label and download the corrected information. This feedback could later be used to improve the model.

# Downloadable Reports

The annotated parking image and a slot-by-slot CSV report can be downloaded for documentation, analysis, or record keeping.

# Professional User Interface

The Streamlit dashboard uses a structured and responsive design. Information is organised into metrics, tabs, charts, status cards, and reports so that the application is easy to use.

# How the Application Works

The user uploads a parking-lot image in JPG, JPEG, or PNG format.

The application reads the predefined coordinates of the parking spaces.

It crops each parking space from the main image.

Each crop is resized to the format required by the model.

The trained model calculates the probability that the space is occupied.

The application classifies the space as Available or Occupied.

The predictions are drawn on the uploaded image.

The dashboard calculates parking availability and utilisation.

The system generates a congestion level and recommendation.

The user can inspect or download the final information.

# Dataset

The project is designed to use the PKLot parking dataset. PKLot contains labelled images of empty and occupied parking spaces recorded under different conditions, including sunny, cloudy, and rainy weather.

Before training, the dataset is organised into three sections:

Training data, used to teach the model

Validation data, used to monitor and improve the model during development

Testing data, used to evaluate the completed model

Using images from different weather and lighting conditions helps the model learn more realistic visual variations.

# Users of the Application

ParkVision AI could be useful for:

Drivers looking for available parking

Parking-facility operators

Shopping centres and commercial buildings

Schools, universities, and office campuses

Hospitals and public facilities

Smart-city traffic-management teams

Urban planners studying parking demand

# Practical Applications

The system could be connected to a fixed parking camera and used to provide regular availability updates. With further development, its information could be displayed on entrance boards, websites, mobile applications, or control-room dashboards.

The parking data could also help operators understand which areas become full most often and when additional parking management is required.

# Benefits

Reduces the need for manual parking inspection

Helps drivers locate available spaces more quickly

Can reduce unnecessary movement inside parking facilities

Presents parking information in a clear visual format

Supports better use of existing parking capacity

Demonstrates a lower-hardware alternative to installing a sensor in every space

Produces useful reports for parking management

Can be expanded for larger smart-city systems

# Limitations

ParkVision AI is a prototype and currently has several limitations:

It requires a fixed camera position and a matching parking-layout file.

It cannot automatically identify new parking-space boundaries in an unrelated image.

Changing the camera angle can cause the predefined parking coordinates to become incorrect.

Strong shadows, glare, darkness, rain, or poor image quality may affect predictions.

Vehicles that are only partly visible may be more difficult to classify.

One vehicle blocking another may create visual uncertainty.

Accuracy depends on the amount, balance, and variety of the training data.

A model trained on one type of parking environment may not perform equally well in every location.

The present version analyses uploaded images rather than a continuous live video stream.

Low-confidence predictions may still require human checking.

These limitations do not make the system unusable, but they define the situations in which its output should be interpreted carefully.

# Future Improvements

The project could be improved in the following ways:

Add automatic parking-space detection for unfamiliar parking areas

Support live CCTV and video-stream analysis

Allow users to create or edit parking layouts inside the application

Train the model with more night, rain, shadow, and low-light images

Add multiple-camera support for larger parking facilities

Use several consecutive video frames to produce more stable predictions

Store historical parking information in a database

Display daily and weekly utilisation trends

Predict future parking demand using historical data

Add alerts when a parking area is nearly full

Connect the system to a mobile application or digital entrance display

Include accessible-parking and reserved-space identification

Use verified user corrections to retrain and improve the model

Add number-plate detection where legally and ethically appropriate

# Social and Environmental Impact

Parking searches contribute to unnecessary vehicle movement. By helping drivers identify available capacity more quickly, a developed version of ParkVision AI could reduce travel time inside parking facilities, lower fuel consumption, and decrease avoidable emissions.

The system can also help parking operators use existing spaces more efficiently. However, any real camera deployment must respect privacy, surveillance, and data-retention requirements.

# Conclusion

ParkVision AI demonstrates how computer vision and deep learning can be applied to a practical urban problem. The application does more than identify whether parking spaces are available or occupied. It converts the predictions into a visual parking map, utilisation statistics, congestion information, recommendations, and downloadable reports.

Although the current version depends on a fixed camera and predefined parking spaces, it provides a strong foundation for a larger intelligent parking-management system. With live-camera support, automatic space detection, additional training data, and historical analytics, the project could be developed into a useful smart-city solution.



streamlit - https://iadai201-2505204--phreyashethwala-ulexplxinsemmybur7ymtr.streamlit.app/
