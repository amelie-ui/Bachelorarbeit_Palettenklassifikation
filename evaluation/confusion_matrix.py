# evaluation/confusion_matrix.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import DATA, PATHS


def plot_confusion_matrix(y_true, y_pred, model_label: str):
    """
    Universelle Konfusionsmatrix – funktioniert für Keras und TFLite.

    Args:
        y_true:      wahre Labels (numpy array)
        y_pred:      vorhergesagte Labels (numpy array)
        model_label: z.B. 'baseline_keras' oder 'baseline_int8'
    """
    class_names = DATA['classes']

    cm            = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Konfusionsmatrix – {model_label}', fontsize=13)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Absolut')
    axes[0].set_xlabel('Vorhergesagte Klasse')
    axes[0].set_ylabel('Tatsächliche Klasse')

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Normalisiert')
    axes[1].set_xlabel('Vorhergesagte Klasse')
    axes[1].set_ylabel('Tatsächliche Klasse')

    plt.tight_layout()
    output_path = PATHS['plots'] / f'confusion_matrix_{model_label}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Gespeichert: {output_path}')
    plt.show()