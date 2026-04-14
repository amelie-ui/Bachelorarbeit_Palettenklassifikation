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
    """F1-Heatmaps im Konfusionsmatrix-Stil:
       - 2x nach Gruppe (Baseline / Augmentation)
       - 3x nach Quantisierung (FP32 / FP16 / INT8)
    """

    f1_cols = [c for c in df.columns if c.startswith('F1 ')]
    if not f1_cols:
        return

    LABEL_MAP   = {'A_PALLET': 'A', 'B_PALLET': 'B', 'C_PALLET': 'C'}
    quant_order = {'fp32': 0, 'fp16': 1, 'int8': 2}

    def _prepare(df_sub, y_label_fn):
        """Matrix + Labels aus einem gefilterten Sub-DataFrame."""
        heat = df_sub.set_index('Modell')[f1_cols].copy()
        heat.columns = [c.replace('F1 ', '') for c in heat.columns]
        heat.columns = [LABEL_MAP.get(c, c) for c in heat.columns]
        heat = heat[[c for c in heat.columns if 'macro' not in c.lower()]]
        values      = heat.values.astype(float)
        col_labels  = list(heat.columns)
        row_labels  = [y_label_fn(m) for m in heat.index]
        return values, col_labels, row_labels

    def _save_heatmap(values, col_labels, row_labels, title, filename):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(values, cmap='Blues', vmin=0, vmax=1)

        ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=2)
        ax.tick_params(which='minor', bottom=False, left=False)

        ax.set_xticks(range(len(col_labels)))
        ax.set_yticks(range(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        ax.set_xlabel('Klasse')
        ax.set_ylabel('')
        ax.set_title(title)

        thresh = 0.5
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f'{values[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if values[i, j] > thresh else 'black',
                        fontsize=12)

        plt.tight_layout()
        out = PATHS['plots'] / filename
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f'Gespeichert: {out}')
        plt.close()

    # --- 1. Gruppe: Baseline / Augmentation ---
    groups = {
        'Baseline':     df[df['Modell'].str.contains('baseline')],
        'Augmentation': df[df['Modell'].str.contains('augmentation')],
    }

    for group_name, df_group in groups.items():
        if df_group.empty:
            continue

        df_group = df_group.copy()
        df_group['_sort'] = df_group['Modell'].apply(
            lambda x: quant_order.get(x.split('_')[-1], 99)
        )
        df_group = df_group.sort_values('_sort').drop('_sort', axis=1)

        values, col_labels, row_labels = _prepare(
            df_group,
            y_label_fn=lambda m: m.split('_')[-1].upper()  # FP32, FP16, INT8
        )
        _save_heatmap(
            values, col_labels, row_labels,
            title=f'Klassenweise F1-Scores – {group_name}',
            filename=f'f1_heatmap_{group_name.lower()}.png'
        )

    # --- 2. Gruppe: FP32 / FP16 / INT8 ---
    for quant in ['fp32', 'fp16', 'int8']:
        df_q = df[df['Modell'].str.endswith(quant)].copy()
        if df_q.empty:
            continue

        # Reihenfolge: baseline vor augmentation
        df_q['_sort'] = df_q['Modell'].apply(lambda x: 0 if 'baseline' in x else 1)
        df_q = df_q.sort_values('_sort').drop('_sort', axis=1)

        values, col_labels, row_labels = _prepare(
            df_q,
            y_label_fn=lambda m: 'Baseline' if 'baseline' in m else 'Augmentation'
        )
        _save_heatmap(
            values, col_labels, row_labels,
            title=quant.upper(),               # ← nur "FP32" / "FP16" / "INT8"
            filename=f'f1_heatmap_{quant}.png'
        )

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