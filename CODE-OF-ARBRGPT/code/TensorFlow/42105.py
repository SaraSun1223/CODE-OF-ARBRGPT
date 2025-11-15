import tensorflow as tf

tf.quantization.quantize_and_dequantize(input=[2.5, 2.5], input_min=[0,0], input_max=[1,1], axis=10)