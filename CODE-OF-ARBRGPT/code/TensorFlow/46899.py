
import tensorflow as tf

# Reproducing the issue with tf.range
tf.range(start=-1e+38, limit=1)

# Reproducing the issue with tf.ragged.range
tf.ragged.range(starts=1e+38)
