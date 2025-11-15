import mxnet as mx

# Set NumPy behavior for MXNet NDArray
mx.npx.set_np()

# Create a 1D array with one element
a = mx.np.arange(1)

# Try slicing beyond the array length (this will raise an error in MXNet)
try:
    print(a[1:])
except IndexError as e:
    print("IndexError occurred:", e)

# Convert to NumPy array and slice again (should work correctly)
print(a.asnumpy()[1:])

# Final value of a
print(a)
