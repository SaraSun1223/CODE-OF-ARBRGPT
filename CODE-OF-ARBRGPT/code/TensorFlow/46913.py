import tensorflow as tf
import numpy as np

tf.keras.layers.RepeatVector(n=9223372036854775807)(np.ones((3, 1)))