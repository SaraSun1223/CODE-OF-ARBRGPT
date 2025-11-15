import tensorflow as tf
import numpy as np

tf.summary.create_file_writer(logdir='', flush_millis=np.ones((1,2)))