# Bachelorarbeit: Palettenklassifikation

Image-Klassifikation für Paletten-Erkennung mit MobileNetV2 und TensorFlow.

## Quick Start

### Installation
```bash
python -m venv abyss
source abyss/bin/activate
pip install -r requirements.txt
```

### Jetson Nano Setup
Siehe [JETSON_SETUP.md](JETSON_SETUP.md) für ARM-spezifische TFLite-Konfiguration.

## Projekt-Struktur

```
├── config.py              # Pfade & Hyperparameter
├── training/              # Modell-Training
│   ├── model.py          # MobileNetV2 Architektur
│   ├── train_baseline.py # Baseline-Training
│   └── train_augmentation.py
├── conversion/           # Quantisierung (FP16, INT8)
├── evaluation/           # Metriken & Grad-CAM
├── visualization/        # Plots & Vergleiche
├── data/                 # Train/Test-Daten
├── jetson/              # Jetson Nano Deployment
│   ├── demo.py          # Live-Inferenz mit Kamera
│   ├── benchmark.py     # Performance-Tests
│   └── test.py          # GPIO-Test
└── notebooks/           # Jupyter Analysen
```

## Befehle

```bash
python run_training.py         # Modell trainieren
python run_evaluation.py       # Metriken & Grad-CAM
python run_conversion.py       # .tflite-Conversion
python jetson/demo.py          # Live-Inferenz (Jetson)
```

## Modelle

- **Baseline:** MobileNetV2 (224×224, 3 Klassen)
- **Augmented:** Mit Augmentations-Pipeline
- **Formate:** Keras (.keras), TFLite (FP32/FP16/INT8)

## Requirements

- Python 3.9+
- TensorFlow 2.15.0
- Siehe `requirements.txt` und `requirements-jetson.txt`

