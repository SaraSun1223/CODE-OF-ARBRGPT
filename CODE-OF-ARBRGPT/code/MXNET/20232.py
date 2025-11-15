from mxnet import np, npx

npx.set_np()
data = np.array([[1,2,3.], [4.,5.,6]])

np.apply_along_axis(lambda x: x.mean(), axis=1, arr=data)