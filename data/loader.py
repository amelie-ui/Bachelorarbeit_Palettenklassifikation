import tensorflow as tf
from config import DATA, PATHS

def load_train_val_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        PATHS['dataset'],
        validation_split=DATA['validation_split'],
        subset='training',
        seed=DATA['seed'],
        image_size=DATA['img_size'],
        batch_size=DATA['batch_size']
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        PATHS['dataset'],
        validation_split=DATA['validation_split'],
        subset='validation',
        seed=DATA['seed'],
        image_size=DATA['img_size'],
        batch_size=DATA['batch_size']
    )

    return train_ds, val_ds

def load_test_dataset(batch_size=1):
    """Lädt den isolierten Testdatensatz."""
    test_ds = tf.keras.utils.image_dataset_from_directory(
        PATHS['dataset_test'],
        image_size=DATA['img_size'],
        batch_size=batch_size,
        shuffle=False
    )
    return test_ds