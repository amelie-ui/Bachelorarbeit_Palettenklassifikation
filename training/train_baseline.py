from config import DATA, TRAINING, PATHS
from training.model import build_model
import tensorflow as tf
from data.loader import load_train_val_datasets

def train_baseline():

    train_ds, val_ds = load_train_val_datasets()
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(245).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    num_classes = len(DATA['classes'])
    model, base_model = build_model(num_classes)

    model.compile(
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = TRAINING['learning_rate']),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=TRAINING['patience'],
        restore_best_weights=True
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(PATHS['models'] / 'baseline.keras'),
        monitor='val_loss',
        save_best_only=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TRAINING['epochs'],
        callbacks=[early_stop, checkpoint]
    )

    print(f"\nModell gespeichert: {PATHS['models'] / 'baseline.keras'}")
    return model, history


if __name__ == '__main__':
    train_baseline()