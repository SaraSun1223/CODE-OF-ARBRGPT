import tensorflow as tf

with tf.Session() as sess:
    a = tf.zeros((2,2), tf.uint64)
    b = tf.zeros((2,2), tf.uint64)
    sess.run(tf.stack([a, b]))