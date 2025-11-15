import sys

import tensorflow as tf

import tensorflow.compat.v1 as v1





g = v1.Graph()

with g.as_default():

    # Create a bunch of variables that are captured in custom_gradient

    captured = [tf.Variable(float(i)) for i in range(1, 20)]



    @tf.custom_gradient

    def FuncMult(x):

        def GradMult(*dys, variables=None):

            return (

                4. * sum(captured) * dys[0],

                [(i + 1) * x * y for i in range(len(variables))]

            )



        return x * sum(captured), GradMult



    x = tf.Variable(6.)

    y = FuncMult(x)

    grad = tf.gradients(y, [x])



graph_def = g.as_graph_def(add_shapes=True)

with open(f"tf_graph.pbtxt.{sys.argv[1]}", "w") as f:

    f.write(str(graph_def))