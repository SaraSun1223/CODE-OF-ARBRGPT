import mxnet as mx



x = mx.random.randn(2, 2)

labels = mx.nd.ones(2)

loss = mx.gluon.loss.CosineEmbeddingLoss()



print(loss(x, x, labels))  # works fine

loss.hybridize()

print(loss(x, x, labels))  # shape issue