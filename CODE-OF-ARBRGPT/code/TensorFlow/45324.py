import tensorflow as tf

tfd = tf.data.Dataset.from_tensor_slices(tf.random.normal((2, 200, 200, 3)))
image_6 = tf.random.normal((200, 200, 3))

def test(image_6):

  central_fraction = tf.divide(102, tf.shape(image_6)[0])
  image_7 = tf.image.central_crop(image_6, central_fraction)

  return image_7

test(image_6)