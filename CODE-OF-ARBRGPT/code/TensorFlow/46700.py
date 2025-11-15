import tensorflow as tf
import numpy as np

# Reproduce the issue by calling tf.math.reduce_prod with large values in keepdims
tf.math.reduce_prod(input_tensor=1, keepdims=np.array([63600, 1], dtype=np.float16))