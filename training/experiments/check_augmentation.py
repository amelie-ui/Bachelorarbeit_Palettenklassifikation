import matplotlib.pyplot as plt
import tensorflow as tf
from data.loader import load_train_val_datasets
from training.train_augmentation import build_augmentation
from config import DATA


def visualize_augmentation():
    # 1. Daten laden (nur Training)
    train_ds, _ = load_train_val_datasets()
    class_names = DATA['classes']

    # 2. Augmentierungs-Layer initialisieren
    augment = build_augmentation()

    # 3. Ein Batch (z.B. 4 Bilder) nehmen
    for images, labels in train_ds.take(1):
        plt.figure(figsize=(12, 10))

        for i in range(4):
            # Originalbild
            img = images[i]
            label = class_names[labels[i]]

            # 4. Mehrfache Augmentierung desselben Bildes simulieren
            for j in range(3):
                augmented_img = augment(tf.expand_dims(img, 0), training=True)[0]

                plt.subplot(4, 3, i * 3 + j + 1)
                plt.imshow(augmented_img.numpy().astype("uint8"))
                plt.title(f"{label} (Var {j + 1})")
                plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    visualize_augmentation()