# CardioScanPro 
Anomaly detection in ECG recordings using machine learning 
https://isuru623-cardioscanpro.hf.space

Overview


This web app is designed to assist healthcare professionals in detecting anomalies in ECG (Electrocardiogram) reports. It leverages state-of-the-art deep learning models, including two Convolutional Neural Networks (CNNs) based on the ResNet architecture and a Recurrent Neural Network (RNN) with LSTM layers. The app analyzes ECG data provided in the form of .mat and .hea files, where .mat files contain waveform data and .hea files provide metadata about the waves.

The primary goal of this project is to enhance the accuracy and efficiency of ECG anomaly detection, helping medical practitioners make more informed decisions and provide timely care to patients.

Features


Anomaly Detection: The app employs deep learning models to identify anomalies, irregularities, or potential health risks in ECG data.
Support for Multiple File Formats: You can upload both .mat files (waveform data) and .hea files (metadata) to ensure comprehensive ECG analysis.
State-of-the-Art Models: The models used in this project are trained on a diverse dataset, resulting in robust detection capabilities.
User-Friendly Interface: The web-based user interface makes it easy for healthcare professionals to upload ECG files and receive real-time anomaly detection results.
Customizable Thresholds: The app provides options to customize detection thresholds, allowing for fine-tuning based on specific use cases.

Mainly the training part has been done in the notebooks


Usage

Upload ECG files: Select a .mat file containing waveform data and a corresponding .hea file with metadata. Ensure the format follows the expected structure.


Review Results: The app will provide real-time feedback on detected anomalies in the ECG data.

