# run_training.py
from training.train_baseline import train_baseline
from training.train_augmentation import train_augmentation
from visualization.plot_training import (
    plot_training_history,
    plot_baseline_vs_augmentation
)


def run_training():

    # ── 1. Baseline ───────────────────────────────────────
    print('\n BASELINE')
    model_b, history_b = train_baseline()
    plot_training_history(history_b, 'baseline_e-2')

    print('\n AUGMENTATION_3')
    model_a, history_a = train_augmentation()
    plot_training_history(history_a, 'augmentation_all_e-2')

    # ── 3. Vergleich ──────────────────────────────────────
    print('\n VERGLEICH')
    plot_baseline_vs_augmentation(history_b, history_a)


if __name__ == '__main__':
    run_training()