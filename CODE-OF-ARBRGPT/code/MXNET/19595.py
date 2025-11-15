import mxnet as mx

import numpy as np

mx.npx.set_np()



net = mx.gluon.nn.Embedding(input_dim=32768, output_dim=512)

net.initialize()

net.save_parameters('temp.params')

original_weight = net.collect_params()['weight'].data().asnumpy()

loaded_params = mx.npx.load('temp.params')

loaded_weight = loaded_params['weight'].asnumpy()

np.testing.assert_allclose(original_weight, loaded_weight, 1E-4, 1E-4)