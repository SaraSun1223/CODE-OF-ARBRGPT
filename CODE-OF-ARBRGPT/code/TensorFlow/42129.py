import tensorflow as tf
tf.image.crop_and_resize(image=tf.zeros((2,1,1,1)), boxes=[[1.0e+40, 0,0,0]], box_indices=[1], crop_size=[1,1])