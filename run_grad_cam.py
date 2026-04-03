# evaluation/run_grad_cam.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import DATA, PATHS
from data.loader import load_test_dataset
from evaluation.grad_cam import (
    compute_grad_cam,
    classify_test_images,
    plot_grad_cam_row,
    LABEL_MAP,
)


def analyze_attention_shift(heatmap_b, heatmap_a, threshold=0.6):
    """
    Berechnet statistische Unterschiede zwischen zwei Heatmaps.

    Args:
        heatmap_b: Grad-CAM der Baseline (0.0 - 1.0)
        heatmap_a: Grad-CAM der Augmentation (0.0 - 1.0)
        threshold: Ab welchem Wert ein Pixel als 'Fokus-Region' gilt
    """
    mask_b = (heatmap_b > threshold).astype(np.float32)
    mask_a = (heatmap_a > threshold).astype(np.float32)

    intersection = np.logical_and(mask_b, mask_a).sum()
    union = np.logical_or(mask_b, mask_a).sum()
    iou = intersection / (union + 1e-8)

    def get_centroid(mask):
        coords = np.argwhere(mask)
        return coords.mean(axis=0) if len(coords) > 0 else np.array([112, 112])

    c_b = get_centroid(mask_b)
    c_a = get_centroid(mask_a)
    dist = np.linalg.norm(c_b - c_a)

    return iou, dist


def run_grad_cam_analysis():
    print("── Starte systematische Grad-CAM Analyse ──")

    # 1. Modelle und Daten laden
    baseline  = tf.keras.models.load_model(str(PATHS['models'] / 'baseline.keras'))
    aug_model = tf.keras.models.load_model(str(PATHS['models'] / 'augmentation.keras'))

    test_ds     = load_test_dataset(batch_size=1)
    class_names = DATA['classes']

    # 2. Bulk-Klassifikation
    print("Klassifiziere Testbilder...")
    images, y_true_b, y_pred_b, conf_b = classify_test_images(baseline, test_ds)
    _,      y_true_a, y_pred_a, conf_a = classify_test_images(aug_model, test_ds)

    # 3. Indizes für Analyse-Kategorien
    wrong_b    = np.where(y_true_b != y_pred_b)[0]
    low_conf_b = np.argsort(conf_b)[:3]

    # 4. Fehlklassifikationen der Baseline
    print(f"Erzeuge Heatmaps für {len(wrong_b)} Fehler der Baseline...")
    for idx in wrong_b:
        true_display = LABEL_MAP.get(class_names[y_true_b[idx]],
                                     class_names[y_true_b[idx]])
        pred_display = LABEL_MAP.get(class_names[y_pred_b[idx]],
                                     class_names[y_pred_b[idx]])

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(
            f'Wahre Klasse: {true_display} | '
            f'Baseline (oben): {pred_display} ({conf_b[idx]:.2%})',
            fontsize=11
        )

        heatmap = compute_grad_cam(baseline, images[idx], y_pred_b[idx])
        plot_grad_cam_row(
            axes, images[idx], heatmap,
            class_names[y_true_b[idx]],
            class_names[y_pred_b[idx]],
            conf_b[idx],
            model_name='Baseline'
        )

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = PATHS['grad_cam'] / f'error_baseline_idx{idx}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Gespeichert: {out_path}")

    # 5. Vergleich Baseline vs. Augmentation (Grenzfälle)
    print("Erzeuge Vergleichs-Heatmaps für Grenzfälle...")
    for idx in low_conf_b:
        hm_b = compute_grad_cam(baseline,  images[idx], y_pred_b[idx])
        hm_a = compute_grad_cam(aug_model, images[idx], y_pred_a[idx])

        iou, dist = analyze_attention_shift(hm_b, hm_a)
        print(f"Index {idx}: IoU = {iou:.4f}, Distanz = {dist:.2f} px")

        true_display   = LABEL_MAP.get(class_names[y_true_b[idx]],
                                       class_names[y_true_b[idx]])
        pred_display_b = LABEL_MAP.get(class_names[y_pred_b[idx]],
                                       class_names[y_pred_b[idx]])
        pred_display_a = LABEL_MAP.get(class_names[y_pred_a[idx]],
                                       class_names[y_pred_a[idx]])

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(
            f'Wahre Klasse: {true_display} | '
            f'Baseline (oben): {pred_display_b} ({conf_b[idx]:.2%}) | '
            f'Augmentation (unten): {pred_display_a} ({conf_a[idx]:.2%})',
            fontsize=11
        )

        # Zeile 1: Baseline
        plot_grad_cam_row(
            axes[0], images[idx], hm_b,
            class_names[y_true_b[idx]],
            class_names[y_pred_b[idx]],
            conf_b[idx],
            model_name='Baseline'
        )

        # Zeile 2: Augmentation
        plot_grad_cam_row(
            axes[1], images[idx], hm_a,
            class_names[y_true_a[idx]],
            class_names[y_pred_a[idx]],
            conf_a[idx],
            model_name='Augmentation'
        )

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = PATHS['grad_cam'] / f'comparison_lowconf_idx{idx}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Gespeichert: {out_path}")

    print(f"✓ Analyse abgeschlossen. Bilder gespeichert in: {PATHS['grad_cam']}")


if __name__ == '__main__':
    run_grad_cam_analysis()