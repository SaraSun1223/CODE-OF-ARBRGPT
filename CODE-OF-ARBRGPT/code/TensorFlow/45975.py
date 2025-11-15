import tensorflow as tf

# This line will raise an error
tf.debugging.assert_type(tf.constant(0.0), tf_type=(tf.float32,))