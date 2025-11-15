import mxnet as mx

import json

import pprint

mx.npx.set_np()

net = mx.gluon.nn.BatchNorm(epsilon=2E-5, axis=2)

net.hybridize()

net.initialize()

a = net(mx.np.ones((10, 3, 5, 5)))

net.export('bnorm', 0)

with open('bnorm-symbol.json') as f:

   dat = json.load(f)

   pprint.pprint(dat)