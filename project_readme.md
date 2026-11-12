# Real-Time Face Detection and Recognition Using OpenCV

## Project Overview

This project implements a real-time face detection and recognition system using Python and OpenCV.

It uses the Haar Cascade Classifier for face detection and the LBPH (Local Binary Patterns Histograms) algorithm for face recognition through a webcam.

## Technologies Used

- Python
- OpenCV
- NumPy
- Haar Cascade Classifier
- LBPH Face Recognizer

## Features

- Real-time face detection using webcam
- Facial dataset collection
- Face recognition model training
- Recognition of registered individuals
- Unknown face identification

## Project Workflow

1. Collect face images using the webcam.
2. Detect faces using Haar Cascade.
3. Preprocess and store face images.
4. Train the LBPH recognition model.
5. Recognize faces in real time.
6. Display the person's name or Unknown.

## Installation

Clone the repository and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Execution

1. **Collect face images:**
   ```bash
   python collect_faces.py
   ```

2. **Train the model:**
   ```bash
   python train_model.py
   ```

3. **Run face recognition:**
   ```bash
   python recognize_faces.py
   ```

## Team Members

| Name | Roll Number |
| :--- | :--- |
| Anuj Kumar Sharma | 20233070 |
| Anshul Verma | 20233065 |
| Adarsh Kumar | 20233018 |
| Abhinav Yadav | 20233011 |
| Aditya Kumar | 20233025 |

## Disclaimer

This project is developed for educational purposes as part of an Image Processing and Computer Vision project. It is not intended for production-grade security or authentication.