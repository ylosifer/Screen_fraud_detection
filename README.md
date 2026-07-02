# Screen Fraud Detection

## Overview
Detect whether an uploaded image is:
- 📷 Real camera photo
- 🖥️ Photo of a screen

## Model
- EfficientNet-B0 (Transfer Learning)
- PyTorch

## Dataset
- Custom dataset collected manually
- Real camera photos
- Photos of laptop/monitor screens
- Different lighting, brightness, and viewing angles

## Results

Test Accuracy: 91.11%

Precision: 93.02%

Recall: 88.89%

F1 Score: 90.91%

Latency: ~25 ms/image

## Run

Train

python train.py

Evaluate

python evaluate.py

Predict

python predict.py image.jpg
