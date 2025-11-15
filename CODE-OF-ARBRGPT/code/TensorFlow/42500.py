import tensorflow as tf

N = 100
M = 4

distribute_strategy = tf.distribute.MirroredStrategy()

def op():
  data = tf.random.uniform((N,))
  partitions = tf.random.uniform((N,), maxval=M, dtype=tf.int32)
  return tf.dynamic_partition(data, partitions, M)

distribute_strategy.run(op)