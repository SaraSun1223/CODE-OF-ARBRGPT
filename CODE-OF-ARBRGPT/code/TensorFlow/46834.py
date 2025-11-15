import tensorflow as tf
tf.nn.avg_pool1d(input=tf.ones((1,1,1)), strides=1, ksize=0, padding='VALID')

tf.nn.max_pool3d(input=tf.ones((1,1,1,1,1)), strides=1, ksize=0, padding='VALID')