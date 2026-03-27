import tensorflow as tf
from config import PATHS


def convert_fp16(model_name: str = 'baseline'):
    """
    Konvertiert ein Keras-Modell zu TFLite mit FP16-Quantisierung.
    Gewichte: FP32 → FP16 (~50% Größenreduktion)
    Aktivierungen: bleiben FP32

    Args:
        model_name: 'baseline' oder 'augmentation'
    """

    # ── 1. Keras-Modell laden ─────────────────────────────
    keras_path = PATHS['models'] / f'{model_name}.keras'
    model = tf.keras.models.load_model(str(keras_path))
    print(f"Modell geladen: {keras_path}")

    # ── 2. Konverter mit FP16-Konfiguration ───────────────
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimierungsziel: Größe reduzieren
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Gewichte auf FP16 beschränken
    converter.target_spec.supported_types = [tf.float16]

    # ── 3. Konvertierung durchführen ──────────────────────
    tflite_model = converter.convert()

    # ── 4. Datei speichern ────────────────────────────────
    output_path = PATHS['models'] / f'{model_name}_fp16.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Gespeichert: {output_path.name} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    convert_fp16('baseline')
    convert_fp16('augmentation')
