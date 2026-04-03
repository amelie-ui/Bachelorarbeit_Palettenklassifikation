# run_evaluation.py
from config import PATHS
from evaluation.evaluate_tflite import evaluate_tflite
from visualization.plot_comparisons import main as run_plotting


def run_evaluation():

    print('\n═══ TFLite-Evaluation ═══════════════════════════')
    for model_name in ['baseline', 'augmentation']:
        for quant in ['fp32', 'fp16', 'int8']:
            tflite_path = PATHS['models'] / f'{model_name}_{quant}.tflite'
            if not tflite_path.exists():
                print(f'Übersprungen: {tflite_path.name}')
                continue
            evaluate_tflite(model_name, quant)    # nur Inferenz + JSON

    print('\n✓ Inferenz abgeschlossen. Starte grafische Auswertung...')
    run_plotting()


if __name__ == '__main__':
    run_evaluation()