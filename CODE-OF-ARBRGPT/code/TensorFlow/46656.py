import tensorflow as tf
import os
import numpy as np


class EXAMPLE(tf.Module):

    @tf.function(input_signature=[tf.TensorSpec(shape=[100], dtype=tf.float32)])
    def calculate(self, x):
        maxima_ind = tf.where(x > 0.8)
        maxima_ind = tf.gather(maxima_ind, 0, axis=1)

        maxima_ind = tf.cast(maxima_ind, dtype=tf.float32)

        if len(maxima_ind) > 10:
            maxima_ind = tf.cast(maxima_ind, dtype=tf.float32)

        return maxima_ind


to_export = EXAMPLE()
np.random.seed(54)
buffer_size = 100
x1 = tf.convert_to_tensor(np.random.rand(buffer_size).astype('float32'))

solution = to_export.calculate(x1)
print(solution)
models_dir = '/content/model_example/'
tf.saved_model.save(to_export, models_dir)
imported = tf.saved_model.load(models_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(models_dir)  # path to the SavedModel directory
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()
with open(models_dir + 'model_example.tflite', 'wb') as f:
    f.write(tflite_model)