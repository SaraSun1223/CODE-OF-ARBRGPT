import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class MyLayer1(layers.Layer):

  def call(self, inputs):

    segments = tf.constant([0, 0, 0, 1, 1])
    return tf.transpose(
        tf.math.segment_prod(tf.transpose(inputs), segments))

inputs = layers.Input(10)
embed = layers.Embedding(20, 5)(inputs)
output = MyLayer1()(embed)
output = layers.GlobalAveragePooling1D()(output)
model = tf.keras.Model(inputs, output)

X = np.random.randint(20, size=(100, 10))
y = np.random.randn(100,2)

model.compile(loss='mae')
model.fit(X, y)