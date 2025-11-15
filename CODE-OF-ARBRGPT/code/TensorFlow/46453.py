import os



import tensorflow as tf



## Download and extract the zip 

## URL: https://drive.google.com/file/d/1Zxqdnm2iHpJGdUl17cAi-lV7wZ3UhMDA/view



params = tf.experimental.tensorrt.ConversionParams(

    precision_mode='FP32',

    maximum_cached_engines=1,

    minimum_segment_size=5)



converter = tf.experimental.tensorrt.Converter(

    input_saved_model_dir='retinanet-18-640-30x-64-tpu',

    conversion_params=params)

converter.convert()



def input_fn(steps=1):

    for i in range(steps):

        yield (tf.random.uniform([640, 640, 3]), tf.constant(1, dtype=tf.int32))

        

converter.build(input_fn=input_fn)

converter.save('trt')