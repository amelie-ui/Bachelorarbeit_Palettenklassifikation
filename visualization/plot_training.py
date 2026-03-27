# visualization/plot_training.py
import matplotlib.pyplot as plt
import json
from config import PATHS


def plot_training_history(history, model_name: str):
    """
    Plottet Trainings- und Validierungskurven (Accuracy + Loss).

    Args:
        history:    Keras History-Objekt aus model.fit()
        model_name: 'baseline' oder 'augmentation'
    """
    epochs = range(1, len(history.history['accuracy']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Trainingsverlauf – {model_name}', fontsize=13)

    # ── Accuracy ──────────────────────────────────────────
    ax1.plot(epochs, history.history['accuracy'],
             label='Training', color='steelblue')
    ax1.plot(epochs, history.history['val_accuracy'],
             label='Validierung', color='orange')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Loss ──────────────────────────────────────────────
    ax2.plot(epochs, history.history['loss'],
             label='Training', color='steelblue')
    ax2.plot(epochs, history.history['val_loss'],
             label='Validierung', color='orange')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = PATHS['plots'] / f'training_{model_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Gespeichert: {output_path}')
    plt.show()


def plot_baseline_vs_augmentation(history_b, history_a):
    """
    Vergleicht Trainingsverläufe von Baseline und Augmentation.
    Direkt nutzbar als Abbildung in Kap. 5.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Baseline vs. Augmentation – Validierungsverlauf', fontsize=13)

    for ax, metric, title in zip(
        axes,
        ['val_accuracy', 'val_loss'],
        ['Validation Accuracy', 'Validation Loss']
    ):
        epochs_b = range(1, len(history_b.history[metric]) + 1)
        epochs_a = range(1, len(history_a.history[metric]) + 1)

        ax.plot(epochs_b, history_b.history[metric],
                label='Baseline', color='steelblue')
        ax.plot(epochs_a, history_a.history[metric],
                label='Augmentation', color='orange')
        ax.set_title(title)
        ax.set_xlabel('Epoche')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = PATHS['plots'] / 'baseline_vs_augmentation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Gespeichert: {output_path}')
    plt.show()