# conversion/convert_int8.py
import tensorflow as tf
import numpy as np
from config import PATHS, DATA


def get_calibration_dataset(n_images: int = 100):
    """
    Erstellt einen Kalibrierungsdatensatz aus dem Trainingsdatensatz.
    Wird benötigt um Aktivierungsbereiche pro Schicht zu messen.
    Bilder werden identisch zum Training vorverarbeitet: [0,255] → [-1,1]

    Args:
        n_images: Anzahl der Kalibrierungsbilder (Standard: 100)
    """

    dataset = tf.keras.utils.image_dataset_from_directory(
        PATHS['dataset'],
        image_size=DATA['img_size'],
        batch_size=1,
        shuffle=True,
        seed=DATA['seed']
    )

    def representative_dataset():
        for images, _ in dataset.take(n_images):
            image = tf.cast(images, tf.float32)
            # Kein preprocess_input — Modell normalisiert intern
            yield [image]

    return representative_dataset


def convert_int8(model_name: str = 'baseline'):
    """
    Konvertiert ein Keras-Modell zu TFLite mit vollständiger INT8-Quantisierung.
    Gewichte und Aktivierungen werden auf INT8 quantisiert (~75% Größenreduktion).
    Der Kalibrierungsdatensatz bestimmt die Aktivierungsbereiche pro Schicht.

    Args:
        model_name: 'baseline' oder 'augmentation'
    """

    # ── 1. Keras-Modell laden ─────────────────────────────
    keras_path = PATHS['models'] / f'{model_name}.keras'
    model = tf.keras.models.load_model(str(keras_path))
    print(f"Modell geladen: {keras_path}")

    # ── 2. Konverter mit INT8-Konfiguration ───────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = get_calibration_dataset(n_images=100)

    # Vollständige INT8-Quantisierung: Gewichte + Aktivierungen + Ein-/Ausgabe
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.int8

    # ── 3. Konvertierung durchführen ──────────────────────
    print("Kalibrierung läuft – 100 Bilder werden durchlaufen...")
    tflite_model = converter.convert()

    # ── 4. Datei speichern ────────────────────────────────
    output_path = PATHS['models'] / f'{model_name}_int8.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Gespeichert: {output_path.name} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    convert_int8('baseline')
    convert_int8('augmentation')