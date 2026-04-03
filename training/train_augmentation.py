import tensorflow as tf
from config import DATA, TRAINING, PATHS
from training.model import build_model
from tensorflow.keras import layers
from data.loader import load_train_val_datasets

def build_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
        layers.RandomZoom(height_factor=(-0.1, 0.0)),
    ], name='augmentation')

def train_augmentation(experiment_name: str = 'augmentation'):
    train_ds, val_ds = load_train_val_datasets()

    # Augmentierung nur auf Trainingsdaten
    augment = build_augmentation()
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    # Optimierungen nach Augmentierung
    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

    num_classes = len(DATA['classes'])
    model, base_model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(
            #TRAINING['learning_rate']
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience= 5,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(PATHS['models'] / f'{experiment_name}.keras'),
        monitor='val_loss',
        save_best_only=True
    )

    print(f"\nPhase 1: Feature Extraction (/f'{experiment_name})\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TRAINING['epochs'],
        callbacks=[early_stop, checkpoint]
    )

    print(f"\nModell gespeichert: {PATHS['models'] / f'{experiment_name}.keras'}")
    return model, history


if __name__ == '__main__':
    train_augmentation('augmentation')
