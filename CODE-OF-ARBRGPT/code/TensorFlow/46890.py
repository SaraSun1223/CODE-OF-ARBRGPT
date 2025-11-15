
import tensorflow as tf
import numpy as np

# tf.image.resize
tf.image.resize(images=np.ones((5,5,5)), size=[2065374891,1145309325])

# tf.image.resize_with_crop_or_pad
tf.image.resize_with_crop_or_pad(image=np.ones((1,1,1)), target_height=5191549470, target_width=5191549470)

# tf.image.pad_to_bounding_box
tf.image.pad_to_bounding_box(image=np.ones((1,1,1)), target_height=5191549470, target_width=5191549470, offset_height=1, offset_width=1)

# tf.image.extract_glimpse
tf.image.extract_glimpse(input=np.ones((5,5,5,5)), size=[1574700351, 451745106], offsets=np.ones((5,2)))

# tf.keras.backend.resize_images
tf.keras.backend.resize_images(x=np.ones((1,5,3,15)), height_factor=5628955348197345288, width_factor=5628955348197345288, data_format='channels_last')
