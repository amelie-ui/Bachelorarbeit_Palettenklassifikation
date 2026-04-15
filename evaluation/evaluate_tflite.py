# evaluation/evaluate_tflite.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import json
from config import DATA, PATHS
from data.loader import load_test_dataset


def preprocess_for_tflite(image, input_details):
    """
    Bereitet ein Bild für TFLite-Inferenz vor.
    Alle Varianten (FP32, FP16, INT8) erwarten FP32-Eingabe —
    preprocess_input ist im Modell eingebaut.
    """
    image = tf.cast(image, tf.float32).numpy()
    return np.expand_dims(image, axis=0)


def run_tflite_inference(interpreter, image):
    """Führt eine einzelne Inferenz durch (ohne Zeitmessung)."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    return output[0]


def evaluate_tflite(model_name: str = 'baseline', quantization: str = 'fp32'):
    """
    Evaluiert ein TFLite-Modell auf dem Testdatensatz (nur Klassifikationsmetriken).
    """
    tflite_path = PATHS['models'] / f'{model_name}_{quantization}.tflite'
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Modell geladen: {tflite_path.name}")
    print(f"Input dtype:  {input_details[0]['dtype']}")
    print(f"Output dtype: {output_details[0]['dtype']}")

    test_ds = load_test_dataset(batch_size=1)
    class_names = DATA['classes']

    y_true, y_pred, y_prob = [], [], []

    for image, label in test_ds:
        img = image[0]
        processed = preprocess_for_tflite(img, input_details)
        prediction = run_tflite_inference(interpreter, processed)

        y_true.append(label.numpy()[0])
        y_pred.append(np.argmax(prediction))
        y_prob.append(prediction)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    accuracy = np.mean(y_true == y_pred)
    macro_f1 = report['macro avg']['f1-score']

    print(f"\n── Ergebnisse: {model_name}_{quantization} ──────────────")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"Makro-F1:   {macro_f1:.4f}")
    print("\nKlassenweise Metriken:")
    for class_name in class_names:
        r = report[class_name]
        print(f"  {class_name}: Precision={r['precision']:.4f} Recall={r['recall']:.4f} F1={r['f1-score']:.4f}")

    output = {
        'model': model_name,
        'quantization': quantization,  # Nur bei TFLite
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'y_true': y_true.tolist(),  # .tolist() required for JSON serialization
        'y_pred': y_pred.tolist(),
        'report': report,
    }
    output_path = PATHS['metrics'] / f'{model_name}_{quantization}_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nMetriken gespeichert: {output_path}")

    return y_true, y_pred, y_prob


if __name__ == '__main__':
    for model in ['baseline', 'augmentation']:
        for quant in ['fp32', 'fp16', 'int8']:
            evaluate_tflite(model, quant)