import mxnet as mx

import numpy as np



class SoftReLU(mx.gluon.HybridBlock):

    def hybrid_forward(self, F, x):

        x = F.identity(x)

        x = F.Activation(x, act_type='softrelu')

        x = F.identity(x)

        return x



fun_normal = SoftReLU()

fun_hybrid = SoftReLU()

fun_hybrid.hybridize()

inp = 100



for f_name, fun in [('Normal', fun_normal), ('Hybrid', fun_hybrid)]:

    for c_name, ctx in [('CPU', mx.cpu()), ('GPU', mx.gpu())]:

        for dtype in [np.float16, np.float32, np.float64]:

            data = mx.nd.array([inp], ctx=ctx, dtype=dtype)

            result = fun(data)

            i_dtype = np.dtype(dtype).name

            o_dtype = np.dtype(result.dtype).name

            print(f"{f_name} on context {c_name} ({i_dtype} => {o_dtype}) SoftReLU({inp}) = {result.asnumpy().item()}")