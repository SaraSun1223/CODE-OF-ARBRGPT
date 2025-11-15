import tensorflow as tf

print(tf.__version__)

conv2d = tf.keras.layers.Conv2D(16, (3, 3), use_bias=False)

@tf.function
def func(x):
  return conv2d(x)

converter = tf.lite.TFLiteConverter.from_concrete_functions([func.get_concrete_function(tf.TensorSpec([None, 32, 32, 3]))])
with open("nightly.tflite", "wb") as f:
    f.write(converter.convert())