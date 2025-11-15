import mxnet as mx



data = mx.nd.random.randn(1, 1, 8, 8)



conv_t = mx.gluon.nn.Conv2DTranspose(in_channels=1, channels=2, kernel_size=3, strides=1)

conv_t.initialize()



print(conv_t)

print("++++++++++++")

print(conv_t.summary(data))