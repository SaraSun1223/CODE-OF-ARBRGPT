import tensorflow as tf

pool_size = [2, 2, 0]

layer = tf.keras.layers.MaxPooling3D(strides=1, pool_size=pool_size)

input_tensor = tf.random.uniform([3, 4, 10, 11, 12], dtype=tf.float32)

res = layer(input_tensor) # crash