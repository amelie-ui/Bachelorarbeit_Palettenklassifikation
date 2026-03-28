# run_evaluation.py
import json
from config import PATHS
from evaluation.evaluate_keras import evaluate_keras
from evaluation.evaluate_tflite import evaluate_tflite
# Import der Plot-Logik als abschließender Schritt
from visualization.plot_comparisons import main as run_plotting

def run_evaluation():
    """
    Führt die Inferenz für alle Modelle aus und erzeugt die Metrik-JSONs.
    Keine Berechnungen von Zeit/FPS hier, da diese auf dem Jetson Nano erfolgen.
    """

    # ── 1. Keras-Modelle evaluieren ───────────────────────
    # Erzeugt die Metriken für Baseline und Augmentation
    print('\n═══ Keras-Evaluation (Inferenz) ════════════════')
    for model_name in ['baseline', 'augmentation']:
        evaluate_keras(model_name)

    # ── 2. TFLite-Modelle evaluieren ──────────────────────
    # Erzeugt Metriken für alle Quantisierungsstufen
    print('\n═══ TFLite-Evaluation (Inferenz) ═══════════════')
    for model_name in ['baseline', 'augmentation']:
        for quant in ['fp32', 'fp16', 'int8']:
            tflite_path = PATHS['models'] / f'{model_name}_{quant}.tflite'
            if not tflite_path.exists():
                print(f'Übersprungen: {tflite_path.name} (nicht vorhanden)')
                continue

            evaluate_tflite(model_name, quant)

    # ── 3. Übergabe an den Visualisierungshub ─────────────
    # Dies erzeugt nun zentral alle Tabellen, Balkendiagramme und Konfusionsmatrizen
    print('\n✓ Inferenz abgeschlossen. Starte grafische Auswertung...')
    run_plotting()


if __name__ == '__main__':
    run_evaluation()