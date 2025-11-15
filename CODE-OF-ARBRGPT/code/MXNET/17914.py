import mxnet as mx

import numpy as np


class MyModel(mx.gluon.HybridBlock):

    def __init__(self) -> None:
        super().__init__()

        with self.name_scope():
            self.cs = self.params.get_constant("cs", np.arange(10))

    def hybrid_forward(self, F, dummy, cs):
        u = F.broadcast_add(cs, dummy.zeros_like())

        # r = F.broadcast_add(

        #     F.slice_axis(u, axis=-1, begin=0, end=-1),

        #     F.slice_axis(u, axis=-1, begin=1, end=None)

        # ) / 2.0

        r = (

                    F.slice_axis(u, axis=-1, begin=0, end=-1) +

                    F.slice_axis(u, axis=-1, begin=1, end=None)

            ) / 2.0

        return r


def main():
    ctx = mx.Context('gpu')

    # ctx = mx.Context('cpu')

    model = MyModel()

    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

    print("hybridizing");
    model.hybridize()

    dummy = mx.nd.array(

        [

            np.ones(10),

            np.ones(10),

            np.ones(10),

        ],

        ctx=ctx

    )

    with mx.autograd.record():
        out = model(dummy)

        print(out.asnumpy())


if __name__ == '__main__':
    main()
