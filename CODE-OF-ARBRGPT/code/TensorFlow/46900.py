
import tensorflow as tf

# First example that crashes
tf.strings.substr(input='abc', len=1, pos=[1,-1])

# Second example that crashes
tf.strings.substr(input='abc', len=1, pos=[1,2])
