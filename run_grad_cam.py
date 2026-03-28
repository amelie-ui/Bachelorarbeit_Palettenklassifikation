# evaluation/run_grad_cam.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import DATA, PATHS
from data.loader import load_test_dataset
from evaluation.grad_cam import (
    compute_grad_cam,
    classify_test_images,
    plot_grad_cam_row
)


def analyze_attention_shift(heatmap_b, heatmap_a, threshold=0.6):
    """
    Berechnet statistische Unterschiede zwischen zwei Heatmaps.

    Args:
        heatmap_b: Grad-CAM der Baseline (0.0 - 1.0)
        heatmap_a: Grad-CAM der Augmentation (0.0 - 1.0)
        threshold: Ab welchem Wert ein Pixel als 'Fokus-Region' gilt
    """
    # Binäre Masken erstellen (Wo schaut das Modell wirklich hin?)
    mask_b = (heatmap_b > threshold).astype(np.float32)
    mask_a = (heatmap_a > threshold).astype(np.float32)

    # Intersection over Union (IoU) berechnen
    intersection = np.logical_and(mask_b, mask_a).sum()
    union = np.logical_or(mask_b, mask_a).sum()
    iou = intersection / (union + 1e-8)

    # Schwerpunkte (Centroids) berechnen
    def get_centroid(mask):
        coords = np.argwhere(mask)
        return coords.mean(axis=0) if len(coords) > 0 else np.array([112, 112])

    c_b = get_centroid(mask_b)
    c_a = get_centroid(mask_a)

    # Euklidische Distanz der Schwerpunkte (in Pixeln)
    dist = np.linalg.norm(c_b - c_a)

    return iou, dist


def run_grad_cam_analysis():
    print("── Starte systematische Grad-CAM Analyse ──")

    # 1. Modelle und Daten laden
    # Nutze die Pfade aus deiner Config für maximale Konsistenz
    baseline = tf.keras.models.load_model(str(PATHS['models'] / 'baseline.keras'))
    # Hier den Namen deines besten Augmentierungs-Modells einsetzen
    aug_model = tf.keras.models.load_model(str(PATHS['models'] / 'augmentation_rotation_brightness_contrast.keras'))

    test_ds = load_test_dataset(batch_size=1)
    class_names = DATA['classes']

    # 2. Bulk-Klassifikation
    print("Klassifiziere Testbilder...")
    images, y_true_b, y_pred_b, conf_b = classify_test_images(baseline, test_ds)
    _, y_true_a, y_pred_a, conf_a = classify_test_images(aug_model, test_ds)

    # 3. Indizes für Analyse-Kategorien extrahieren
    # Fehlklassifikationen
    wrong_b = np.where(y_true_b != y_pred_b)[0]

    # Grenzfälle (niedrigste Konfidenz)
    low_conf_b = np.argsort(conf_b)[:3]

    # 4. Heatmaps für Fehlklassifikationen generieren
    print(f"Erzeuge Heatmaps für {len(wrong_b)} Fehler der Baseline...")
    for idx in wrong_b:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        heatmap = compute_grad_cam(baseline, images[idx], y_pred_b[idx])

        plot_grad_cam_row(
            axes, images[idx], heatmap,
            class_names[y_true_b[idx]],
            class_names[y_pred_b[idx]],
            conf_b[idx]
        )

        out_path = PATHS['grad_cam'] / f'error_baseline_idx{idx}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Direkter Vergleich (Baseline vs. Augmentation)
    print("Erzeuge Vergleichs-Heatmaps für Grenzfälle...")
    for idx in low_conf_b:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        # Heatmaps für beide Modelle berechnen
        hm_b = compute_grad_cam(baseline, images[idx], y_pred_b[idx])
        hm_a = compute_grad_cam(aug_model, images[idx], y_pred_a[idx])

        # NEU: Attention Shift berechnen
        iou, dist = analyze_attention_shift(hm_b, hm_a)
        print(f"Index {idx}: IoU = {iou:.4f}, Distanz = {dist:.2f} px")

        # Zeile 1: Baseline
        plot_grad_cam_row(axes[0], images[idx], hm_b, class_names[y_true_b[idx]],
                          class_names[y_pred_b[idx]], conf_b[idx])
        axes[0, 0].set_ylabel("Baseline")

        # Zeile 2: Augmentation
        plot_grad_cam_row(axes[1], images[idx], hm_a, class_names[y_true_a[idx]],
                          class_names[y_pred_a[idx]], conf_a[idx])
        axes[1, 0].set_ylabel("Augmentation")

        # Titel mit Metriken ergänzen
        fig.suptitle(f'Vergleich Index {idx} | IoU: {iou:.4f} | Distanz: {dist:.2f}px', fontsize=12)

        out_path = PATHS['grad_cam'] / f'comparison_lowconf_idx{idx}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✓ Analyse abgeschlossen. Bilder gespeichert in: {PATHS['grad_cam']}")


if __name__ == '__main__':
    run_grad_cam_analysis()