import mxnet as mx

from mxnet import gluon

from mxnet.gluon import nn

import gluoncv as gcv

class NetEncoder(nn.SymbolBlock):

    def __init__(self, **kwargs):

        base_network = gcv.model_zoo.resnet50_v1(pretrained=False)

        outputs = ['stage1_activation2', 'stage2_activation3', 'stage3_activation5',

                            'stage4_activation2']



        inputs, outputs, params = gcv.nn.feature._parse_network(

            base_network, outputs, ['data'], pretrained=False, ctx=mx.cpu(), **kwargs)

        super(NetEncoder, self).__init__(outputs, inputs, params=params)

    

class Foo(nn.HybridBlock):

    def __init__(self):

        super(Foo, self).__init__()

        self.features = NetEncoder()



    def hybrid_forward(self, F, x):

        y = self.features(x)

        return y



a = mx.nd.random.uniform(shape=(1,3,224,224), ctx=mx.gpu(0))



f = Foo()

f.collect_params().initialize()

f.hybridize()

f.collect_params().reset_ctx(mx.gpu(0))

b = f(a)

print([x.shape for x in b])