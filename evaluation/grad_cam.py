# evaluation/grad_cam.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from config import DATA, PATHS

LAYER_NAME = 'out_relu'

LABEL_MAP = {
    'A_PALLET': 'Klasse A',
    'B_PALLET': 'Klasse B',
    'C_PALLET': 'Klasse C',
}


def compute_grad_cam(model, image, class_index):
    base_model = model.get_layer('mobilenetv2_1.00_224')
    last_conv_layer = base_model.get_layer(LAYER_NAME)

    last_conv_layer_model = tf.keras.Model(
        base_model.inputs, last_conv_layer.output
    )

    # Layer-Namen dynamisch aus dem Modell holen
    pooling_layer = next(
        l for l in model.layers
        if 'global_average_pooling' in l.name
    )
    dropout_layer = next(
        l for l in model.layers
        if 'dropout' in l.name
    )
    dense_layer = next(
        l for l in model.layers
        if 'dense' in l.name
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    x = pooling_layer(x)
    x = dropout_layer(x)
    x = dense_layer(x)
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


def overlay_heatmap(image, heatmap, alpha=0.6):
    colormap = cm.get_cmap('jet')
    heatmap_rgba = colormap(heatmap)

    # image ist jetzt uint8 numpy array
    img = image if isinstance(image, np.ndarray) else image.numpy()
    img = np.clip(img, 0, 255).astype(np.float32)

    heatmap_rgb = heatmap_rgba[..., :3] * 255.0
    a = (heatmap * alpha)[..., np.newaxis]
    overlay = (1 - a) * img + a * heatmap_rgb

    return np.clip(overlay, 0, 255).astype(np.uint8)


def plot_grad_cam_row(axes, image, heatmap, true_label, pred_label, conf,
                      model_name=None):
    overlay = overlay_heatmap(image, heatmap)

    # image ist uint8 numpy array — kein .numpy() nötig
    img = image if isinstance(image, np.ndarray) else image.numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)

    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title('Grad-CAM Overlay')
    axes[1].axis('off')

    if model_name:
        axes[0].set_ylabel(model_name, fontsize=11, labelpad=8)


def classify_test_images(model, test_ds):
    images_list = []
    y_true      = []
    y_pred      = []
    confidences = []

    for image, label in test_ds:
        img  = image[0]
        pred = model.predict(tf.expand_dims(img, 0), verbose=0)[0]

        # Direkt als uint8 speichern — verhindert matplotlib float-Problem
        img_uint8 = np.clip(img.numpy(), 0, 255).astype(np.uint8)

        images_list.append(img_uint8)
        y_true.append(label.numpy()[0])
        y_pred.append(np.argmax(pred))
        confidences.append(np.max(pred))

    return (
        images_list,
        np.array(y_true),
        np.array(y_pred),
        np.array(confidences)
    )


