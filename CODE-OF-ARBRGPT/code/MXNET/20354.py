import mxnet as mx
from mxnet.gluon import nn
mx.npx.set_np()

class FCEltwise(nn.HybridBlock):
    def __init__(self, use_bias, flatten, **kwargs):
        super(FCEltwise, self).__init__(**kwargs)
        self.fc = nn.Dense(units=64, use_bias=use_bias, flatten=flatten,
                         weight_initializer=None)

    def forward(self, x):
        fc_out = self.fc(x)
        out = mx.np.square(fc_out)
        return out

attrs = {'fc': {'with_eltwise': 'true'}}
net = FCEltwise(True, True)

net.initialize()
net.hybridize(static_alloc=False, static_shape=False)
data = mx.np.random.uniform(size=(64, 4, 10, 10), dtype='float32', ctx=mx.current_context())
net(data)
sym, params = net.export(None)

sym_sg = sym.optimize_for('MKLDNN', dedup_subgraph=True, skip_infer=True)
for name, attrs in attrs.items():
    if len(attrs):
        found = False
        for k, v in sym_sg.attr_dict().items():
            if k.find('sg_mkldnn_fully_connected') != -1:
                found = True
                for attr_name, attr_value in attrs.items():
                    assert v[attr_name].lower() == attr_value.lower()
        assert found