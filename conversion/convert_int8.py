# conversion/convert_int8.py
import tensorflow as tf
import numpy as np
from config import PATHS, DATA


def get_calibration_dataset(n_images: int = 100):
    """
    Erstellt einen Kalibrierungsdatensatz aus dem Trainingsdatensatz.
    Wird benötigt um Aktivierungsbereiche pro Schicht zu messen.

    Args:
        n_images: Anzahl der Kalibrierungsbilder (Standard: 100)
    """

    # Trainingsdatensatz laden (ohne Split, nur für Kalibrierung)
    dataset = tf.keras.utils.image_dataset_from_directory(
        PATHS['dataset'],
        image_size=DATA['img_size'],
        batch_size=1,          # Einzelbilder für präzise Messung
        shuffle=True,
        seed=DATA['seed']
    )

    # Generator-Funktion: LiteRT erwartet einen callable der Batches liefert
    def representative_dataset():
        for images, _ in dataset.take(n_images):
            # Preprocessing wie im Training: [0,255] → [-1,1]
            image = tf.cast(images, tf.float32)
            yield [image]

    return representative_dataset


def convert_int8(model_name: str = 'baseline'):
    """
    Konvertiert ein Keras-Modell zu TFLite mit INT8-Quantisierung.
    Gewichte: FP32 → INT8 (~75% Größenreduktion)
    Aktivierungen: FP32 → INT8 (erfordert Kalibrierung)

    Args:
        model_name: 'baseline' oder 'augmentation'
    """

    # ── 1. Keras-Modell laden ─────────────────────────────
    keras_path = PATHS['models'] / f'{model_name}.keras'
    model = tf.keras.models.load_model(str(keras_path))
    print(f"Modell geladen: {keras_path}")

    # ── 2. Konverter mit INT8-Konfiguration ───────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimierungsziel: maximale Kompression
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Kalibrierungsdatensatz übergeben
    # → LiteRT misst Aktivierungsbereiche pro Schicht
    converter.representative_dataset = get_calibration_dataset(n_images=100)

    # Vollständige INT8-Quantisierung erzwingen:
    # Eingang und Ausgang des Modells ebenfalls auf INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
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