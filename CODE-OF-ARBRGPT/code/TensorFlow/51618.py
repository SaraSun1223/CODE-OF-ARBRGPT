import tensorflow as tf

import numpy as np

x = np.arange(9).reshape([1,3,3,1])

res = tf.image.extract_glimpse(x, size=[1023, -63], offsets=[1023, 63], centered=False, normalized=False) # Crash