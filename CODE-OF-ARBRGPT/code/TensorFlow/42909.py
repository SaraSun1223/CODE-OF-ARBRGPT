import tensorflow as tf



x = tf.ones(5)

with tf.GradientTape() as g:

    g.watch(x)

    y = tf.math.reduce_prod(x)



grad = g.gradient(y, x)