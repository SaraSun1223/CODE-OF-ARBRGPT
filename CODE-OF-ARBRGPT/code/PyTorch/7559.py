import torch

i = slice(0)

x = torch.Tensor(range(8)).view(2, 2, 2)

print(x[:, i])  # Wrong result: tensor([ 0.,  4.]) - Correct: tensor([])

print(x[0, i])  # Correct: tensor([])

try:
    print(x[i, 0])  # Error: RuntimeError: dimension out of range
except RuntimeError as e:
    print(e)