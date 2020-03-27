import tensorflow as tf

def rgb_example(x, y):
    y_a, y_b, y_c, y_d, kde, kappa = y

    y_a = tf.one_hot(tf.cast(y_a, tf.uint8), depth=3).numpy()
    y_b = tf.one_hot(tf.cast(y_b, tf.uint8), depth=2).numpy()
    y_c = tf.one_hot(tf.cast(y_c, tf.uint8), depth=2).numpy()
    y_d = tf.one_hot(tf.cast(y_d, tf.uint8), depth=7).numpy()

    feature = {}
    feature['x'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[x.tobytes()]))
    feature['y_a'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[y_a.tobytes()]))
    feature['y_b'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[y_b.tobytes()]))
    feature['y_c'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[y_c.tobytes()]))
    feature['y_d'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[y_d.tobytes()]))
    feature['kappa'] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[kappa.tobytes()]))

    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_rgb_example(record, instance_shape):
    features = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y_a': tf.io.FixedLenFeature([], tf.string),
        'y_b': tf.io.FixedLenFeature([], tf.string),
        'y_c': tf.io.FixedLenFeature([], tf.string),
        'y_d': tf.io.FixedLenFeature([], tf.string),
        'kappa': tf.io.FixedLenFeature([], tf.string),
    }

    image_features = tf.io.parse_single_example(record, features=features)

    x = tf.io.decode_raw(image_features.get('x'), tf.float32)
    x = tf.reshape(x, instance_shape)

    y_a = tf.io.decode_raw(image_features.get('y_a'), tf.float32)
    y_a = tf.reshape(y_a, (3,))
    y_b = tf.io.decode_raw(image_features.get('y_b'), tf.float32)
    y_b = tf.reshape(y_b, (2,))
    y_c = tf.io.decode_raw(image_features.get('y_c'), tf.float32)
    y_c = tf.reshape(y_c, (2,))
    y_d = tf.io.decode_raw(image_features.get('y_d'), tf.float32)
    y_d = tf.reshape(y_d, (7,))
    kappa = tf.io.decode_raw(image_features.get('kappa'), tf.float32)

    return x, y_a, y_b, y_c, y_d, kappa
