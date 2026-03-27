# evaluation/confusion_matrix.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data.loader import load_test_dataset
from config import DATA, PATHS


def plot_confusion_matrix(model_name: str = 'baseline'):
    """
    Erstellt Konfusionsmatrix für ein Keras-Modell auf dem Testdatensatz.

    Args:
        model_name: 'baseline' oder 'augmentation'
    """

    # ── 1. Modell und Testdatensatz laden ─────────────────
    keras_path = PATHS['models'] / f'{model_name}.keras'
    model = tf.keras.models.load_model(str(keras_path))
    test_ds = load_test_dataset(batch_size=DATA['batch_size'])
    class_names = DATA['classes']

    # ── 2. Vorhersagen sammeln ────────────────────────────
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── 3. Konfusionsmatrix berechnen ─────────────────────
    cm = confusion_matrix(y_true, y_pred)

    # Normalisierte Variante (Anteile statt absolute Zahlen)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # ── 4. Visualisierung ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Konfusionsmatrix – {model_name}', fontsize=13)

    # Absolut
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title('Absolut')
    axes[0].set_xlabel('Vorhergesagte Klasse')
    axes[0].set_ylabel('Tatsächliche Klasse')

    # Normalisiert
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_title('Normalisiert')
    axes[1].set_xlabel('Vorhergesagte Klasse')
    axes[1].set_ylabel('Tatsächliche Klasse')

    plt.tight_layout()

    output_path = PATHS['plots'] / f'confusion_matrix_{model_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Gespeichert: {output_path}")
    plt.show()


if __name__ == '__main__':
    plot_confusion_matrix('baseline')
    plot_confusion_matrix('augmentation')