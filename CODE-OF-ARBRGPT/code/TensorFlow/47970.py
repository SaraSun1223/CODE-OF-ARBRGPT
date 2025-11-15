import tensorflow as tf

def floordiv(x, y):
    # x // y
    return tf.math.floordiv(x, y)

@tf.function
def floordiv_tffn(x, y):
    # x // y
    return tf.math.floordiv(x, y)

@tf.function(experimental_compile=True)
def floordiv_compiled(x, y):
    # x // y
    return tf.math.floordiv(x, y)

x, y = tf.constant([0., 0.1, 0.9]), 1.
print(floordiv(x, y))
print(floordiv_tffn(x, y))
print(floordiv_compiled(x, y))