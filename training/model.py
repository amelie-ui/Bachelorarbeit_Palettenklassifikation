import tensorflow as tf
from config import PATHS, MODEL
from tensorflow.keras import layers
from config import DATA


#Constants
Image_size = (224, 224)
Image_shape = Image_size + (3,)

def build_model(num_classes: int):

    #Backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape = Image_shape,
        alpha = 1.0,
        include_top = False,
        weights = 'imagenet',
#       pooling = 'avg',
    )
    base_model.trainable = False

    # Modell Pipeline
    inputs = tf.keras.Input(shape = Image_shape)
    # Normalisierung auf [-1, 1] (MobileNetV2-Erwartung)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # Feature Extraction (training=False: BatchNorm im Inferenzmodus)
    x = base_model(x, training=False)

    # Klassifikationskopf --> hier muss ich nochmal schauen wie ich das begründe!
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(MODEL['dropout'])(x)
    #will ich Dense???
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    base_model = model.get_layer('mobilenetv2_1.00_224')
    for layer in base_model.layers:
        print(layer.name)

    return model, base_model
if __name__ == '__main__':
    model, base_model = build_model(len(DATA['classes']))
