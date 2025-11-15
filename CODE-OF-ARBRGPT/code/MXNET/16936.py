from mxnet import np, npx

npx.set_np()

print((np.ones((2, 0, 2)) * np.ones((2,))).sum(-1))