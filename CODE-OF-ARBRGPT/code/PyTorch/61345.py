import torch



# init module

class MyModule(torch.nn.Module):

    def __init__(self):

        super(MyModule, self).__init__()

        self.register_buffer("rb", torch.randn(1, 1, 3, 1, 1)) 

        # you can run with the workaround

        # self.rb = torch.randn(1, 1, 3, 1, 1)



    def forward(self, x):

        x += self.rb[0]

        return x



torch_model = MyModule().eval()



# torch.onnx.export

torch.onnx.export(torch_model,

        torch.randn(1, 3, 224, 224),

        "./tmp.onnx",

        input_names=["inputs"],

        output_names=["outputs"],

        dynamic_axes={"inputs": {0: "batch", 2: "height", 3: "width"}, "outputs": {0: "batch", 1: "class", 2: "height", 3: "width"}},

        opset_version=11,

        export_params=True)



# onnxruntime

import os

import numpy as np

import onnxruntime

from onnxruntime.datasets import get_example



onnx_model = get_example(os.path.join(os.getcwd(), "tmp.onnx"))

sess = onnxruntime.InferenceSession(onnx_model)

inputs = np.random.randn(1, 3, 224, 224).astype(np.float32)

onnx_out = sess.run(None, {"inputs": inputs})