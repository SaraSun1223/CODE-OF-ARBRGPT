import tensorflow as tf

with tf.device("/gpu:0"):
  t_int32 = tf.identity(tf.constant([0, 0, 1], dtype=tf.int32))
  t_float32 = tf.identity(tf.constant([0, 0, 1], dtype=tf.float32))

  dlp_int32 = tf.experimental.dlpack.to_dlpack(t_int32)
  dlp_float32 = tf.experimental.dlpack.to_dlpack(t_float32)

  ft_int32 = tf.experimental.dlpack.from_dlpack(dlp_int32)
  ft_float32 = tf.experimental.dlpack.from_dlpack(dlp_float32)

  print(ft_int32.shape)
  print(ft_float32.shape)  # This works for both

  print(ft_int32) # This crash
  print(ft_float32)