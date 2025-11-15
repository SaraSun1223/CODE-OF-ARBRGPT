
import mxnet
import gluoncv

net = gluoncv.model_zoo.get_model('ssd_512_resnet50_v1_voc', \
    pretrained=False,
    pretrained_base=False,
    norm_layer=None,
    use_bn=False,
    norm_kwargs= None)
net.initialize()
net.cast("float16")                                                               # loses efficacy


one = mxnet.nd.zeros((1,3,512,512), dtype="float16")       # meet error

net(one)