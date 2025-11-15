import argparse



import mxnet as mx

import numpy as np



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Demonstrate an MXNet incompatibility with PySide2.")

    parser.add_argument("--import-pyside", dest="import_pyside", action="store_true", default=False)

    parser.add_argument("--create-app", dest="create_app", action="store_true", default=False)

    parser.add_argument("--scalar", type=float, default=0.6)

    args = parser.parse_args()

    if args.import_pyside:

        from PySide2.QtCore import QCoreApplication



        if args.create_app:

            app = QCoreApplication()



    tensor = mx.nd.full(shape=(375, 600, 3), val=255).astype(np.int32)

    tensor = tensor.astype(np.float32)

    scalar = args.scalar

    print("All of the following operations should yield the same result:")

    print("Multiply numpy ndarray by scalar: ", (tensor.asnumpy() * scalar).max())

    print("Multiply MXNet NDArray by MXNet NDArray: ", (tensor * mx.nd.array([[[scalar]]])).max().asscalar())

    print("Multiply MXNet NDArray by float: ", (tensor * scalar).max().asscalar())