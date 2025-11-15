import tensorflow as tf

x = tf.random.uniform((5, 3))
w = tf.zeros((5, 1))

result = tf.nn.weighted_moments(x, axes=0, frequency_weights=w)

print(result)