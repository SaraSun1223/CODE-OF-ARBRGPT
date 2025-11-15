from mxnet import np, npx

npx.set_np()

np.random.normal(0, np.ones((2,2)))

np.random.gumbel(np.ones((2,2)))

np.random.logistic(np.ones((2,2)))

np.random.gumbel(np.ones((2,2)), np.ones((2,2)))