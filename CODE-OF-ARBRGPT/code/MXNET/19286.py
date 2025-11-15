from mxnet import np, npx, gluon, use_np

npx.set_np()


@use_np
class TestModel(gluon.HybridBlock):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0]


model = TestModel()

model.initialize()

model.hybridize()

model(np.zeros((2, 2, 4, 0, 128)))
