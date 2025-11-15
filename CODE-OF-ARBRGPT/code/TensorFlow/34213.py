from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

import tensorflow_datasets as tfds
import tensorflow as tf

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["192.168.1.191:12345", "192.168.1.193:12345"]
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

tfds.disable_progress_bar()

BUFFER_SIZE = 10000
BATCH_SIZE = 64


def make_datasets_unbatched():
    # Scaling MNIST data from (0, 255] to (0., 1.]
    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255
        return image, label

    datasets, info = tfds.load(name='mnist',
                               with_info=True,
                               as_supervised=True)

    return datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)


def build_and_compile_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model


# single_worker_model = build_and_compile_cnn_model()
# single_worker_model.fit(x=train_datasets, epochs=3)

NUM_WORKERS = 2
# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64,
# and now this becomes 128.
GLOBAL_BATCH_SIZE = 16 * NUM_WORKERS
with strategy.scope():
    # Creation of dataset, and model building/compiling need to be within
    # `strategy.scope()`.
    train_datasets = make_datasets_unbatched().batch(GLOBAL_BATCH_SIZE)
    multi_worker_model = build_and_compile_cnn_model()
multi_worker_model.fit(x=train_datasets, epochs=5)