from mxnet import np, npx

npx.set_np()



A = np.array([1, 2, 3, 4])

np.pad(A, pad_width=(1, 2))