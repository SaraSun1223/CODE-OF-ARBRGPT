import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def main():
    inputs = layers.Input(shape=(None,))
    x = tf.signal.stft(inputs, 512, 20, pad_end=True)
    model = keras.Model(inputs=inputs, outputs=x)
    signals = tf.constant(np.random.rand(2, 511))
    print(model(signals))
    print('All done.')


if __name__ == '__main__':
    main()