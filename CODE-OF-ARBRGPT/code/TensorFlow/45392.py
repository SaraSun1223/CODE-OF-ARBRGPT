import tensorflow as tf

t = tf.SparseTensor(indices=[[0, 0, 0, 0, 0, 0]], values=[0.0], dense_shape=[4096, 4096, 4096, 4096, 4096, 4096])

tf.sparse.reorder(t)