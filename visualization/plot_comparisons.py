# visualization/plot_comparisons.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import PATHS, DATA
from evaluation.confusion_matrix import plot_confusion_matrix

def load_all_metrics():
    """
    Liest alle TFLite-Metrik-JSONs ein.
    Keras-JSONs werden übersprungen – fp32 ist inhaltlich identisch.
    """
    data = []

    for filepath in sorted(PATHS['metrics'].glob('*_metrics.json')):
        # Keras-Dateien überspringen → fp32 ist identisch
        if 'keras' in filepath.name:
            continue

        with open(filepath) as f:
            d = json.load(f)

        quant       = d.get('quantization', 'fp32')
        model_type  = d.get('model', 'unknown')
        model_label = f"{model_type}_{quant}"
        report      = d['report']
        macro       = report.get('macro avg', {})

        row = {
            'Modell':             model_label,
            'Accuracy':           d.get('accuracy'),
            'Precision (macro)':  macro.get('precision'),
            'Recall (macro)':     macro.get('recall'),
            'F1 (macro)':         d.get('macro_f1', macro.get('f1-score')),
            'y_true':             d.get('y_true'),
            'y_pred':             d.get('y_pred'),
        }

        # Klassenweise F1-Scores
        for class_name, values in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                row[f'F1 {class_name}'] = values['f1-score']

        data.append(row)

    if not data:
        print('Keine Metrik-Dateien gefunden.')
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Sortierung: baseline vor augmentation, fp32 → fp16 → int8
    quant_order = {'fp32': 0, 'fp16': 1, 'int8': 2}
    df['_sort'] = df['Modell'].apply(lambda x: (
        0 if 'baseline' in x else 1,
        quant_order.get(x.split('(')[-1].rstrip(')'), 99)
    ))
    df = df.sort_values('_sort').drop('_sort', axis=1).reset_index(drop=True)
    return df


def plot_metrics_bar(df):
    """Balkendiagramm: Accuracy/Precision/Recall/F1 je Variante."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x     = range(len(df))
    width = 0.2

    ax.bar([i - 1.5*width for i in x], df['Accuracy'],          width, label='Accuracy',          color='steelblue')
    ax.bar([i - 0.5*width for i in x], df['Precision (macro)'], width, label='Precision (macro)', color='lightcoral')
    ax.bar([i + 0.5*width for i in x], df['Recall (macro)'],    width, label='Recall (macro)',    color='lightgreen')
    ax.bar([i + 1.5*width for i in x], df['F1 (macro)'],        width, label='F1 (macro)',        color='orange')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Klassifikationsleistung im Vergleich')
    ax.set_xticks(list(x))
    ax.set_xticklabels(df['Modell'], rotation=30, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PATHS['plots'] / 'metrics_comparison.png', dpi=150)
    print('Gespeichert: metrics_comparison.png')
    plt.show()

def plot_f1_heatmap(df):
    """Heatmap: klassenweise F1-Scores je Variante."""

    f1_cols = [c for c in df.columns if c.startswith('F1 ')]
    if not f1_cols:
        return

    heat = df.set_index('Modell')[f1_cols].copy()
    heat.columns = [c.replace('F1 ', '') for c in heat.columns]

    # Klassennamen umbenennen
    LABEL_MAP = {
        'A_PALLET': 'A',
        'B_PALLET': 'B',
        'C_PALLET': 'C',
    }
    heat.columns = [LABEL_MAP.get(c, c) for c in heat.columns]

    # Makro-Spalte entfernen falls vorhanden
    heat = heat[[c for c in heat.columns if 'macro' not in c.lower()]]

    plt.figure(figsize=(8, max(4, len(df) * 0.6)))

    ax = sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        linecolor='white',
        cbar=False,          # Farbskala entfernt
        vmin=0,
        vmax=1,
        annot_kws={"size": 9, "weight": "bold"}
    )

    plt.title('Klassenweise F1-Scores', fontsize=14, weight='bold')
    ax.set_xlabel('')   # Achsenbeschriftungen entfernt
    ax.set_ylabel('')

    plt.xticks(rotation=0, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(PATHS['plots'] / 'f1_heatmap.png', dpi=200)
    print('Gespeichert: f1_heatmap.png')
    plt.close()

def generate_confusion_matrices(df):
    """
    Konfusionsmatrizen aus gespeicherten y_true/y_pred.
    Einzige Stelle wo Konfusionsmatrizen erstellt werden.
    """
    print('\n=== Konfusionsmatrizen ===')
    for _, row in df.iterrows():
        if row['y_true'] is None or row['y_pred'] is None:
            print(f"  Übersprungen (keine Rohdaten): {row['Modell']}")
            continue

        label = row['Modell'].replace(' ', '_').replace('(', '').replace(')', '')
        plot_confusion_matrix(
            np.array(row['y_true']),
            np.array(row['y_pred']),
            label
        )


def main():
    df = load_all_metrics()
    if df.empty:
        return

    # 1. Tabelle (→ Kap. 5)
    main_cols = ['Modell', 'Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)']
    print('\n=== Metriken ===\n')
    print(df[main_cols].to_markdown(index=False, floatfmt='.4f'))

    # 2. Balkendiagramm
    plot_metrics_bar(df)

    # 3. F1-Heatmap
    plot_f1_heatmap(df)

    # 4. Konfusionsmatrizen – nur hier, nicht in run_evaluation
    generate_confusion_matrices(df)


if __name__ == '__main__':
    main()