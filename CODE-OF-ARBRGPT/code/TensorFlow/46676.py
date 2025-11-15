import tensorflow as tf
import numpy as np

# Create a numpy array with dtype uint32
a = np.array([3], dtype=np.uint32)

# Execute tf.raw_ops.PopulationCount
tf.raw_ops.PopulationCount(x=a)