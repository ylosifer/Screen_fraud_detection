# Screen Fraud Detection

## Overview

A lightweight computer vision system that detects whether an image is:
- 📷 A real camera photograph
- 🖥️ A photograph of a digital screen

The model is built using **EfficientNet-B0** with transfer learning in PyTorch and is designed for fast, on-device inference.

---

## Model

- EfficientNet-B0 (Transfer Learning)
- PyTorch
- AdamW Optimizer
- Early Stopping
- ReduceLROnPlateau Scheduler

---

## Dataset

A custom dataset collected manually containing:
- Real camera photographs
- Photos captured from laptop/monitor screens
- Multiple lighting conditions
- Different screen brightness levels
- Various indoor and outdoor scenes

---

## Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **93.33%** |
| **Test Accuracy** | **91.11%** |
| Precision | 93.02% |
| Recall | 88.89% |
| F1 Score | 90.91% |
| Inference Latency | ~25 ms/image |

---

## Run

### Train

```bash
python train.py
```

### Evaluate

```bash
python evaluate.py
```

### Predict

```bash
python predict.py image.jpg
```
