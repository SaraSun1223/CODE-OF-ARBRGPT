from mxnet import np, npx

from mxnet.gluon import HybridBlock



class block1(HybridBlock):

    def hybrid_forward(self, F, x):

        return x + 1



class block2(HybridBlock):

    def forward(self, x):

        return x + 1



net1 = block1()

net1.hybridize()

net2 = block2()

net2.hybridize()



print(type(net1(np.ones((2,2)))))



print(type(net2(np.ones((2,2)))))