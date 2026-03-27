import tensorflow as tf
from config import PATHS


def convert_fp32(model_name: str = 'baseline'):

    keras_path = PATHS['models'] / f'{model_name}.keras'
    model = tf.keras.models.load_model(str(keras_path))
    print(f"Modell geladen: {keras_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    output_path = PATHS['models'] / f'{model_name}_fp32.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Gespeichert: {output_path.name} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    convert_fp32('baseline')
    convert_fp32('augmentation')