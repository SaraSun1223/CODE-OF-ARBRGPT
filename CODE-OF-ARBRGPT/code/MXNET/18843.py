import mxnet as mx



net_fp32 = mx.gluon.model_zoo.vision.resnet34_v2(pretrained=False)

net_fp32.initialize()

net_fp32(mx.nd.zeros((1,3,224,224)))

net_fp32.cast('float64')

net_fp32.hybridize()

data = mx.nd.zeros((1,3,224,224), dtype='float64')

net_fp32(data)

sym_file, params_file = net_fp32.export('test', 0)



sm = mx.sym.load(sym_file)

inputs = mx.sym.var('data', dtype='float64')

net_fp64 = mx.gluon.SymbolBlock(sm, inputs)

net_fp64.load_parameters(params_file)