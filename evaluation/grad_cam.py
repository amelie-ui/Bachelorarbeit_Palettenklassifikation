# evaluation/grad_cam.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from config import DATA, PATHS

LAYER_NAME = 'out_relu'


def compute_grad_cam(model, image, class_index):
    base_model = model.get_layer('mobilenetv2_1.00_224')
    last_conv_layer = base_model.get_layer(LAYER_NAME)

    # Teil 1: MobileNetV2 bis out_relu
    last_conv_layer_model = tf.keras.Model(
        base_model.inputs, last_conv_layer.output
    )

    # Teil 2: Klassifikator neu verdrahtet
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in ['global_average_pooling2d', 'dropout', 'dense']:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.Model(classifier_input, x)

    image_batch = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs = last_conv_layer_model(image_batch)
        tape.watch(conv_outputs)
        preds = classifier_model(conv_outputs)
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis], DATA['img_size']
    ).numpy().squeeze()

    return heatmap


def overlay_heatmap(image, heatmap, alpha=0.4):
    """
    Legt Heatmap als Jet-Colormap über das Originalbild.

    Args:
        image:   Originalbild (224, 224, 3) als uint8
        heatmap: Grad-CAM Heatmap (224, 224) in [0, 1]
        alpha:   Transparenz der Heatmap (0=unsichtbar, 1=vollständig)

    Returns:
        overlay: Überlagertes Bild als uint8
    """
    colormap = cm.get_cmap('jet')
    heatmap_colored = colormap(heatmap)[..., :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    image_uint8 = image.numpy().astype(np.uint8)
    overlay = (1 - alpha) * image_uint8 + alpha * heatmap_colored
    return overlay.astype(np.uint8)


def classify_test_images(model, test_ds):
    """
    Klassifiziert alle Bilder im Testdatensatz.

    Returns:
        images_list:  Liste aller Bilder als Tensoren
        y_true:       Wahre Labels
        y_pred:       Vorhergesagte Labels
        confidences:  Max. Softmax-Wert je Bild
    """
    images_list = []
    y_true      = []
    y_pred      = []
    confidences = []

    for image, label in test_ds:
        img  = image[0]
        pred = model.predict(tf.expand_dims(img, 0), verbose=0)[0]

        images_list.append(img)
        y_true.append(label.numpy()[0])
        y_pred.append(np.argmax(pred))
        confidences.append(np.max(pred))

    return (
        images_list,
        np.array(y_true),
        np.array(y_pred),
        np.array(confidences)
    )


def plot_grad_cam_row(axes, image, heatmap, true_label, pred_label, conf):
    """
    Zeichnet eine Zeile: Original | Heatmap | Overlay.
    Wiederverwendbar für alle Notebook-Zellen.
    """
    overlay = overlay_heatmap(image, heatmap)

    axes[0].imshow(image.numpy().astype('uint8'))
    axes[0].set_title(f'Original\nWahr: {true_label}')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nPred: {pred_label} ({conf:.2%})')
    axes[2].axis('off')