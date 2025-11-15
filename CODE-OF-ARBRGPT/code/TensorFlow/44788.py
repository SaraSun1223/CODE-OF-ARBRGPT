import tensorflow as tf

# The same function: decorated with tf.function (will be executed in graph mode) and
# not decorated (will be executed in eager mode)

@tf.function
def graph_func(x):
    unique_input_ids, idx, counts = tf.raw_ops.UniqueWithCountsV2(x=x, axis=[0])
    tf.print('idx shape', tf.shape(idx), 'idx', idx)
    return x

def eager_func(x):
    unique_input_ids, idx, counts = tf.raw_ops.UniqueWithCountsV2(x=x, axis=[0])
    tf.print('idx shape', tf.shape(idx), 'idx', idx)
    return x

c = tf.constant([[0,0,1],
                 [0,0,1],
                 [0,0,2],
                 [0,0,1]])
_ = graph_func(c)
_ = eager_func(c)