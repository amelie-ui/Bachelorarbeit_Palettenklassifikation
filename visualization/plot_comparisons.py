# visualization/plot_comparisons.py

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import PATHS, DATA
from evaluation.confusion_matrix import plot_confusion_matrix


def load_all_metrics():
    """Liest alle JSONs im metrics-Ordner und bereitet sie für Plots vor."""
    metrics_dir = PATHS['metrics']
    data = []

    for filepath in metrics_dir.glob("*_metrics.json"):
        if filepath.name == "all_results.json":
            continue

        with open(filepath, 'r') as f:
            d = json.load(f)

        # Label erstellen (z.B. 'baseline (int8)' oder 'augmentation (keras)')
        model_type = d.get('model', 'unknown')
        quant = d.get('quantization', 'keras')
        model_label = f"{model_type} ({quant})"

        report = d['report']
        macro = report.get('macro avg', {})

        # Metriken für die Haupttabelle extrahieren
        row = {
            'Modell': model_label,
            'Accuracy': d.get('accuracy'),
            'Precision (macro)': macro.get('precision'),
            'Recall (macro)': macro.get('recall'),
            'F1 (macro)': d.get('macro_f1', macro.get('f1-score')),
            'y_true': d.get('y_true'),  # Rohdaten für Konfusionsmatrix
            'y_pred': d.get('y_pred')
        }

        # Klassenweise F1-Scores für die Heatmap
        for class_name, values in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                row[f'F1 {class_name}'] = values['f1-score']

        data.append(row)

    if not data:
        print("Keine Metrik-Dateien gefunden!")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # Sortierung: Keras -> FP32 -> FP16 -> INT8
    df['SortKey'] = df['Modell'].apply(lambda x:
                                       (0 if '(keras)' in x else 1,
                                        0 if 'baseline' in x else 1,
                                        {'fp32': 0, 'fp16': 1, 'int8': 2}.get(x.split('(')[-1].rstrip(')'), 99)))
    df = df.sort_values('SortKey').drop('SortKey', axis=1)
    return df


def generate_confusion_matrices_from_df(df):
    """Erzeugt Konfusionsmatrizen direkt aus den im DataFrame gespeicherten Listen."""
    print("\n=== Erzeuge Konfusionsmatrizen aus JSON-Daten ===")
    for _, row in df.iterrows():
        if row['y_true'] and row['y_pred']:
            # Konvertierung zurück in Numpy für den Plotter
            plot_confusion_matrix(
                np.array(row['y_true']),
                np.array(row['y_pred']),
                row['Modell'].replace(" ", "_").replace("(", "").replace(")", "")
            )


def plot_metrics_bar(df):
    """Erstellt das vergleichende Balkendiagramm."""
    plt.figure(figsize=(12, 6))
    x = range(len(df))
    width = 0.2

    plt.bar([i - 1.5 * width for i in x], df['Accuracy'], width, label='Accuracy', color='steelblue')
    plt.bar([i - 0.5 * width for i in x], df['Precision (macro)'], width, label='Precision (macro)', color='lightcoral')
    plt.bar([i + 0.5 * width for i in x], df['Recall (macro)'], width, label='Recall (macro)', color='lightgreen')
    plt.bar([i + 1.5 * width for i in x], df['F1 (macro)'], width, label='F1 (macro)', color='orange')

    plt.xlabel('Modellvariante')
    plt.ylabel('Score')
    plt.title('Klassifikationsleistung im Vergleich')
    plt.xticks(x, df['Modell'], rotation=30, ha='right')
    plt.legend(loc='lower right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PATHS['plots'] / 'metrics_comparison.png', dpi=150)
    plt.show()


def main():
    df = load_all_metrics()
    if df.empty: return

    # 1. Tabelle für die Thesis
    main_cols = ['Modell', 'Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)']
    print("\n=== Metriken als Markdown ===\n")
    print(df[main_cols].to_markdown(index=False, floatfmt=".4f"))

    # 2. Balkendiagramm
    plot_metrics_bar(df)

    # 3. Konfusionsmatrizen (jetzt ohne erneutes Modell-Laden!)
    generate_confusion_matrices_from_df(df)


if __name__ == '__main__':
    main()