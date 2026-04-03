# evaluation/confusion_matrix.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from config import DATA, PATHS

LABEL_MAP = {
    'A_PALLET': 'A',
    'B_PALLET': 'B',
    'C_PALLET': 'C',
}

def plot_confusion_matrix(y_true, y_pred, model_label: str):
    """
    Konfusionsmatrix (absolut) für eine Modellvariante.

    Args:
        y_true:      wahre Labels (numpy array)
        y_pred:      vorhergesagte Labels (numpy array)
        model_label: z.B. 'baseline_fp32'
    """
    class_names = [LABEL_MAP.get(c, c) for c in DATA['classes']]
    cm_abs = confusion_matrix(y_true, y_pred)
    cm = cm_abs.astype(float) / cm_abs.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, cmap='Blues')

    # Trennlinien zwischen Zellen
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Vorhergesagte Klasse')
    ax.set_ylabel('Tatsächliche Klasse')
    ax.set_title(f'{model_label}')

    # Zahlenwerte in den Zellen
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.0%}',
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=12)

    plt.tight_layout()
    output_path = PATHS['plots'] / f'confusion_matrix_{model_label}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Gespeichert: {output_path}')
    plt.close()