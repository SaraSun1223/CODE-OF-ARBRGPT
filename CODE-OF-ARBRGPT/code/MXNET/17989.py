import mxnet as mx

from mxnet.gluon import nn, HybridBlock

import numpy as np

import argparse

mx.npx.set_np()



np.random.seed(123)

mx.random.seed(123)





parser = argparse.ArgumentParser(

        description='Grad req bug minimal example')

parser.add_argument('--addto', action='store_true')

parser.add_argument('--hybridize', action='store_true')

parser.add_argument('--nrepeat', type=int, default=5)

args = parser.parse_args()



class Foo(HybridBlock):

    def __init__(self, prefix=None, params=None):

        super().__init__(prefix=prefix, params=params)

        with self.name_scope():

            self.layer = nn.Dense(16)



    def hybrid_forward(self, F, dat):

        out = dat

        for _ in range(args.nrepeat):

            out = self.layer(out)

        return out



foo = Foo()

if args.hybridize:

   foo.hybridize()

foo.initialize(ctx=mx.gpu())



if args.grad_addto:

    for p in foo.collect_params().values():

        p.grad_req = 'add'

foo.collect_params().zero_grad()





dat = mx.np.random.normal(0, 1, (32, 16), ctx=mx.gpu())

og = mx.np.random.normal(0, 1, (32, 16), ctx=mx.gpu())

with mx.autograd.record():

    out = foo(dat)

    loss = (out * og).sum()

    loss.backward()

for k, v in foo.collect_params().items():

    print(k, mx.np.linalg.norm(v.grad()))