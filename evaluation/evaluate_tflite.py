# evaluation/evaluate_tflite.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import json
import time
from config import DATA, PATHS
from data.loader import load_test_dataset


def preprocess_for_tflite(image, input_details):
    """
    Bereitet ein Bild für TFLite-Inferenz vor.
    Bei INT8: Skalierung von FP32 auf INT8 über Quantisierungsparameter.
    Bei FP32/FP16: nur dtype-Konvertierung.
    """
    dtype = input_details[0]['dtype']

    if dtype == np.int8:
        # INT8: Quantisierungsparameter aus dem Modell auslesen
        # scale und zero_point wurden während der Kalibrierung berechnet
        scale, zero_point = input_details[0]['quantization']

        # MobileNetV2 Preprocessing: [0,255] → [-1,1]
        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.cast(image, tf.float32)
        )
        # FP32 → INT8: Formel aus TFLite-Spezifikation
        # quantized = float / scale + zero_point
        image = np.round(image.numpy() / scale + zero_point).astype(np.int8)
    else:
        # FP32/FP16: nur dtype-Konvertierung
        image = tf.cast(image, tf.float32).numpy()

    return np.expand_dims(image, axis=0)    # Batch-Dimension ergänzen


def run_tflite_inference(interpreter, image):
    """
    Führt einen einzelnen Inferenzschritt durch und misst die Zeit.

    Returns:
        prediction: Softmax-Wahrscheinlichkeiten
        duration_ms: Inferenzzeit in Millisekunden
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)

    # Zeitmessung: nur die eigentliche Inferenz
    start = time.perf_counter()
    interpreter.invoke()
    duration_ms = (time.perf_counter() - start) * 1000

    output = interpreter.get_tensor(output_details[0]['index'])

    # Bei INT8: Ausgabe zurück auf FP32 skalieren
    if output_details[0]['dtype'] == np.int8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    return output[0], duration_ms


def evaluate_tflite(model_name: str = 'baseline', quantization: str = 'fp32'):
    """
    Evaluiert ein TFLite-Modell auf dem Testdatensatz.

    Args:
        model_name:   'baseline' oder 'augmentation'
        quantization: 'fp32', 'fp16' oder 'int8'
    """

    # ── 1. Interpreter laden ──────────────────────────────
    tflite_path = PATHS['models'] / f'{model_name}_{quantization}.tflite'
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Modell geladen: {tflite_path.name}")
    print(f"Input dtype:  {input_details[0]['dtype']}")
    print(f"Output dtype: {output_details[0]['dtype']}")

    # ── 2. Testdatensatz laden ────────────────────────────
    test_ds = load_test_dataset(batch_size=1)
    class_names = DATA['classes']

    # ── 3. Vorhersagen + Zeitmessung ─────────────────────
    y_true       = []
    y_pred       = []
    y_prob       = []
    inference_times = []

    for image, label in test_ds:
        img = image[0]  # (1, 224, 224, 3) → (224, 224, 3)
        processed = preprocess_for_tflite(img, input_details)

        # Inferenz
        prediction, duration_ms = run_tflite_inference(interpreter, processed)

        y_true.append(label.numpy()[0])
        y_pred.append(np.argmax(prediction))
        y_prob.append(prediction)
        inference_times.append(duration_ms)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # ── 4. Metriken berechnen ─────────────────────────────
    report   = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )
    accuracy = np.mean(y_true == y_pred)

    # Inferenzzeit-Statistiken
    avg_ms = np.mean(inference_times)
    fps    = 1000 / avg_ms    # Bilder pro Sekunde

    # ── 5. Ausgabe ────────────────────────────────────────
    print(f"\n── Ergebnisse: {model_name}_{quantization} ──────────────")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Ø Inferenzzeit:    {avg_ms:.2f} ms")
    print(f"FPS:               {fps:.1f}")
    print(f"\nKlassenweise Metriken:")
    for class_name in class_names:
        r = report[class_name]
        print(f"  {class_name}: "
              f"Precision={r['precision']:.4f} "
              f"Recall={r['recall']:.4f} "
              f"F1={r['f1-score']:.4f}")

    # ── 6. Ergebnisse speichern ───────────────────────────
    output = {
        'model':        model_name,
        'quantization': quantization,
        'accuracy':     accuracy,
        'avg_ms':       avg_ms,
        'fps':          fps,
        'report':       report,
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