import tensorflow as tf

import numpy as np



@tf.function

def gather_fn(x, indices, axis, batch_dims):

    return tf.gather(x, indices, axis=axis, batch_dims=batch_dims)



# 2-D input with shape (2, 3)

x = tf.constant(np.arange(6).reshape(2, 3), dtype=tf.int32)

# 3-D indices with shape (2, 1, 2)

indices = tf.constant([[[0,1]], [[1,0]]], dtype=tf.int32)

axis = 1

batch_dims = -3

# Eager mode computes correctly

print('Eager gather:', tf.gather(x, indices, axis=axis, batch_dims=batch_dims))

# Error in graph mode

print('Function gather', gather_fn(x, indices, axis=axis, batch_dims=batch_dims))