import tensorflow as tf
import numpy as np


fast_optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3)

slow_optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3 * 1e-9)


@tf.function
def apply_gradients_once(optimizer, grads, vars):
    grads = [grads]
    optimizer.apply_gradients(zip(grads, vars))


def apply_grads(use_fast, grads_per_model, vars):
    for i in range(2):
        if use_fast[i]:
            apply_gradients_once(fast_optimizer, grads_per_model[i], vars[i])
        else:
            apply_gradients_once(slow_optimizer, grads_per_model[i], vars[i])


def compute_loss(w, x, y):
    r = (w * x - y)**2
    r = tf.math.reduce_mean(r)
    return r

def compute_gradients(model):
    with tf.GradientTape() as tape:
        tape.watch(model)
        loss = compute_loss(model, x, y)
    grads = tape.gradient(loss, model)
    return grads


w = [
    tf.Variable(0.0),
    tf.Variable(1.0)]

x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

vars = []
grads = []
for i in range(2):
    vars.append([w[i]])
    grads.append(compute_gradients(w[i]))

apply_grads([True, False], grads, vars)