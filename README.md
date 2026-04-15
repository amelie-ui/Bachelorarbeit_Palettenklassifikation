# DasWirdGut

Image classification system for pallet detection using MobileNetV2 with TensorFlow Keras.

## Setup

### Standard Installation
```bash
python -m venv abyss
source abyss/bin/activate
pip install -r requirements.txt
```

### Jetson Nano / Xavier / Orin
Follow [JETSON_SETUP.md](JETSON_SETUP.md) for ARM-specific TFLite configuration.

## Project Structure

```
├── config.py                  # Global paths & hyperparameters
├── training/                  # Model training & callbacks
│   ├── model.py              # MobileNetV2 architecture
│   ├── train_baseline.py     # Baseline training script
│   └── train_augmentation.py # Augmented dataset training
├── conversion/               # Model quantization (FP16, INT8)
├── evaluation/               # Metrics & Grad-CAM analysis
├── visualization/            # Training plots & comparisons
├── data/                      # Train/test data loaders
├── jetson/                    # Jetson Nano deployments
│   ├── demo.py              # Real-time inference with GPIO
│   ├── benchmark.py         # Performance benchmarking
│   └── test.py              # GPIO hardware testing
└── notebooks/               # Jupyter analysis notebooks
```

## Quick Start

### Train a Model
```bash
python run_training.py
```

### Evaluate & Convert
```bash
python run_evaluation.py  # Keras model metrics
python run_conversion.py  # Generate FP16/INT8 .tflite files
```

### Jetson Deployment
```bash
python jetson/demo.py     # Real-time inference with camera
python jetson/benchmark.py # Performance metrics
```

## Models

- **Baseline:** MobileNetV2 (224×224, 3 classes)
- **Augmented:** Same architecture with augmentation pipeline
- **Formats:** Keras (.keras), TFLite (FP32/FP16/INT8)

## Requirements

- Python 3.9+
- TensorFlow 2.15.0
- See `requirements.txt` for dependencies
- See `requirements-jetson.txt` for ARM-specific packages
