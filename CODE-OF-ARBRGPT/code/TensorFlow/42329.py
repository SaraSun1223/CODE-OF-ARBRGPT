import tensorflow as tf
import numpy as np

# Case 1: Invalid `check_types` parameter
try:
    tf.nest.assert_same_structure(nest1=np.zeros((1)), nest2=tf.ones((1,1,1)), check_types=tf.ones((2)))
except Exception as e:
    print(f"Case 1 Error: {e}")

# Case 2: Invalid `expand_composites` parameter
try:
    tf.nest.assert_same_structure(nest1=np.zeros((1)), nest2=tf.ones((1,1,1)), expand_composites=tf.ones((2)))
except Exception as e:
    print(f"Case 2 Error: {e}")