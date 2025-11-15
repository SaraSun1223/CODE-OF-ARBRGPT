import tensorflow as tf

import gzip



f = open('data.gz', 'wb')

for i in range(10):

        line = str(i) + '\n' + str(i + 100) + '\n'

        f.write(gzip.compress(line.encode('utf-8')))

f.close()



ds = tf.data.TextLineDataset('data.gz', compression_type='GZIP')

for i in ds:

        print(i)