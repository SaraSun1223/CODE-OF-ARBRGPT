import tensorflow as tf
import numpy as np

tf.keras.backend.tile(x=np.ones((1,1,1)), n=[100000000,100000000, 100000000])