import io

import numpy as np

import onnx

import onnxruntime as ort

import torch

import torch.nn.functional as F



from torch import nn





class StackTime(nn.Module):

    __constants__ = ["factor"]

    def __init__(self, factor):

        super().__init__()

        self.factor = int(factor)

    def forward(self, x, x_lens):

        r = torch.transpose(x, 0, 1)

        s = r.shape

        r = F.pad(r, [0, 0, 0, (-s[1]) % self.factor, 0, 0])

        s = r.shape

        rs = [s[0], s[1]//self.factor, s[2]*self.factor]

        r = torch.reshape(r, rs)

        x_lens = torch.ceil(x_lens.float() / self.factor).int()

        return torch.transpose(r, 0, 1), x_lens



stm = StackTime(factor=2)



max_seq_len = 7

batch = 8

hidden = 3



torch.manual_seed(42)

x = torch.rand(max_seq_len, batch, hidden)

x_lens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 7])

sx, sx_lens = stm(x, x_lens)



bitstream = io.BytesIO()

torch.onnx.export(

    model=stm,

    args=(x, x_lens),

    f=bitstream,

    input_names=["x", "x_lens"],

    opset_version=11,

    dynamic_axes={"x": {0: "seq_len", 1: "batch"}, "x_lens": {0: "batch"}})

bitstream_data = bitstream.getvalue()

ort_session = ort.InferenceSession(bitstream_data)

ort_inputs = {"x": x.numpy(), "x_lens": x_lens.numpy()}

ort_outputs = ort_session.run(None, ort_inputs)

ort_sx, ort_sx_lens = ort_outputs



print(x.detach().numpy().shape)  # (7, 8, 3)

print(x_lens.detach().numpy().shape)  # (8,)

print(sx.detach().numpy().shape)  # (4, 8, 6)

print(sx_lens.detach().numpy().shape)  # (8,)

print(ort_sx.shape)  # (3, 8, 6) -- this is different

print(ort_sx_lens.shape)  # (8,)



np.testing.assert_allclose(ort_sx, sx.detach().numpy(), rtol=1e-5, atol=1e-5)

np.testing.assert_allclose(ort_sx_lens, sx_lens.detach().numpy(), rtol=1e-5, atol=1e-5)