
Real-time Image Detection using MobileNetV2
=================================================
This project implements real-time image classification using a webcam
feed and the MobileNetV2 deep learning model pre-trained on ImageNet.

Features :
===========
-   Real-time webcam-based image classification
-   Uses MobileNetV2 for fast and accurate predictions
-   Displays top prediction with confidence percentage
-   Runs directly on your system using OpenCV

Project Structure :
==================
-   Real-time Image Detection.py – Main Python script that performs
    image classification

Technologies Used :
==================
-   Python
-   OpenCV
-   TensorFlow / Keras
-   MobileNetV2 (ImageNet pre-trained)
-   NumPy

How to Run
===========
1.  Install required dependencies:

        pip install tensorflow opencv-python numpy

2.  Run the script:

        python Real-time Image Detection.py

3.  Press ‘q’ to quit the webcam window.

How It Works
=============
1.  Captures frames from the webcam
2.  Preprocesses the image for MobileNetV2
3.  Performs prediction using ImageNet classes
4.  Displays the detected label with confidence

Use Cases
==========
-   Real-time object classification
-   Computer vision learning projects
-   Rapid prototype for ML-based detection systems


Author
Diksha Kolikal
