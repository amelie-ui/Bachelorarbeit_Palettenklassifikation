# evaluation/evaluate_keras.py
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
from config import DATA, PATHS
from data.loader import load_test_dataset


def evaluate_keras(model_name: str = 'baseline'):
    """
    Evaluiert ein Keras-Modell auf dem Testdatensatz.
    Berechnet Accuracy, Precision, Recall, F1 pro Klasse.

    Args:
        model_name: 'baseline' oder 'augmentation'
    """

    # ── 1. Modell und Testdatensatz laden ─────────────────
    keras_path = PATHS['models'] / f'{model_name}.keras'
    model = tf.keras.models.load_model(str(keras_path))
    print(f"Modell geladen: {keras_path}")

    test_ds = load_test_dataset(batch_size=DATA['batch_size'])
    class_names = DATA['classes']

    # ── 2. Vorhersagen sammeln ────────────────────────────
    y_true = []
    y_pred = []
    y_prob = []    # Softmax-Wahrscheinlichkeiten → für Grenzfälle in Grad-CAM

    for images, labels in test_ds:
        predictions = model(images, training=False).numpy()  # direkter Aufruf
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
        y_prob.extend(predictions)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # ── 3. Metriken berechnen ─────────────────────────────
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True    # dict statt String → weiterverarbeitbar
    )

    # Gesamtaccuracy
    accuracy = np.mean(y_true == y_pred)

    # ── 4. Ausgabe ────────────────────────────────────────
    print(f"\n── Ergebnisse: {model_name} ──────────────────")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nKlassenweise Metriken:")
    for class_name in class_names:
        r = report[class_name]
        print(f"  {class_name}: "
              f"Precision={r['precision']:.4f} "
              f"Recall={r['recall']:.4f} "
              f"F1={r['f1-score']:.4f}")

    # ── 5. Ergebnisse speichern ───────────────────────────
    # Als JSON → für spätere Vergleichstabelle in Kap. 5
    output = {
        'model':    model_name,
        'accuracy': accuracy,
        'report':   report,
    }
    output_path = PATHS['metrics'] / f'{model_name}_keras_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nMetriken gespeichert: {output_path}")

    # ── 6. Grenzfälle identifizieren → für Grad-CAM ───────
    # Bilder mit niedrigster Konfidenz (max Softmax-Wert)
    confidence = np.max(y_prob, axis=1)
    low_conf_indices = np.argsort(confidence)[:5]    # 5 unsicherste Bilder

    print(f"\nGrenzfälle (niedrigste Konfidenz):")
    for idx in low_conf_indices:
        print(f"  Index {idx}: "
              f"True={class_names[y_true[idx]]} "
              f"Pred={class_names[y_pred[idx]]} "
              f"Konfidenz={confidence[idx]:.4f}")

    return y_true, y_pred, y_prob


if __name__ == '__main__':
    evaluate_keras('baseline')
    evaluate_keras('augmentation')