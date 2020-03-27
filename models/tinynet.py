import tensorflow as tf

def tinynet(ds, shape, num_outputs,):
    inputs = tf.keras.layers.Input(shape=shape)
    x = tf.keras.layers.Conv2D(
        filters=64//ds,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
    )(inputs)

    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(
        filters=128//ds,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
    )(x)

    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding='same')(x)
    x = tf.keras.layers.Conv2D(
        filters=256//ds,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
    )(x)


    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    outputs = tf.keras.layers.Dense(num_outputs)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model
