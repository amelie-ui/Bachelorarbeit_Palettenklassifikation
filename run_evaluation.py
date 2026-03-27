# run_evaluation.py
import json
import numpy as np
import tensorflow as tf
from tabulate import tabulate
from config import DATA, PATHS
from evaluation.evaluate_keras import evaluate_keras
from evaluation.evaluate_tflite import evaluate_tflite
from evaluation.confusion_matrix import plot_confusion_matrix


def print_comparison_table(results: list):
    """
    Gibt eine Vergleichstabelle aller Modellvarianten aus.
    Direkt nutzbar für Kap. 5.
    """
    headers = [
        'Modell', 'Quantisierung',
        'Accuracy', 'F1 (macro)',
        'Ø ms', 'FPS'
    ]
    rows = []

    for r in results:
        report = r['report']
        f1_macro = report['macro avg']['f1-score']

        rows.append([
            r['model'],
            r.get('quantization', 'keras'),
            f"{r['accuracy']:.4f}",
            f"{f1_macro:.4f}",
            f"{r.get('avg_ms', '-')}",
            f"{r.get('fps', '-')}"
        ])

    print('\n── Vergleichstabelle ─────────────────────────────')
    print(tabulate(rows, headers=headers, tablefmt='grid'))


def run_evaluation():
    results = []

    # ── 1. Keras-Modelle ──────────────────────────────────
    print('\n═══ Keras-Evaluation ═══════════════════════════')
    for model_name in ['baseline', 'augmentation']:
        y_true, y_pred, y_prob = evaluate_keras(model_name)
        plot_confusion_matrix(model_name)

        # Gespeicherte Metriken laden
        metrics_path = PATHS['metrics'] / f'{model_name}_keras_metrics.json'
        with open(metrics_path) as f:
            results.append(json.load(f))

    # ── 2. TFLite-Modelle ─────────────────────────────────
    print('\n═══ TFLite-Evaluation ══════════════════════════')
    for model_name in ['baseline', 'augmentation']:
        for quant in ['fp32', 'fp16', 'int8']:
            tflite_path = PATHS['models'] / f'{model_name}_{quant}.tflite'
            if not tflite_path.exists():
                print(f'Übersprungen (nicht gefunden): {tflite_path.name}')
                continue

            y_true, y_pred, y_prob = evaluate_tflite(model_name, quant)

            metrics_path = PATHS['metrics'] / f'{model_name}_{quant}_metrics.json'
            with open(metrics_path) as f:
                results.append(json.load(f))

    # ── 3. Vergleichstabelle ──────────────────────────────
    print_comparison_table(results)

    # ── 4. Alle Ergebnisse zusammengeführt speichern ──────
    output_path = PATHS['metrics'] / 'all_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAlle Ergebnisse gespeichert: {output_path}')


if __name__ == '__main__':
    run_evaluation()