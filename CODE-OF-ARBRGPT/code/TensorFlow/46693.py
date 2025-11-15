import tensorflow as tf
import numpy as np

# Call tf.keras.backend.reshape with a large shape value
tf.keras.backend.reshape(x=[1], shape=np.array([21943, 45817, 30516, 61760, 38987], dtype=np.uint16))