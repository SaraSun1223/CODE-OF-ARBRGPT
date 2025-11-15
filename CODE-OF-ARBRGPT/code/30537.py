import tensorflow as tf
import numpy as np

do_error = True

with tf.Graph().as_default() as graph:
    my_var = tf.Variable(np.ones(5), use_resource=True)
    with tf.device("/gpu:0" if do_error else None):
        gather = tf.gather(my_var, [0, 2, 4])
    opt_op = tf.train.MomentumOptimizer(0.1, 0.1).minimize(gather)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(opt_op)