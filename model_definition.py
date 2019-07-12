import tensorflow as tf
from config import NUM_CLASSES, image_height, image_width, channels

def create_model():
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(3, 3),
                               activation=tf.keras.activations.relu,
                               input_shape=(image_height, image_width, channels)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               activation=tf.keras.activations.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=64,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(units=NUM_CLASSES,
                              activation=tf.keras.activations.softmax)
    ])
    """
    input_tensor = tf.keras.layers.Input(shape=(image_height, image_width, channels))
    model = tf.keras.applications.MobileNetV2(input_shape=(image_height, image_width, channels), input_tensor=input_tensor, include_top=False)
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    dense = tf.keras.layers.Dense(NUM_CLASSES, name='dense')(avg_pool)
    logits = tf.keras.layers.Activation('softmax', name='logits')(dense)
    model = tf.keras.models.Model(input_tensor, logits)
    model.summary()
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  metrics=['accuracy'])

    return model
