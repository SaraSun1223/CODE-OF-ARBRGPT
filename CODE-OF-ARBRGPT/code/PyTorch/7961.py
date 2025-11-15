import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 初始化模型并设置为评估模式（问题根源）
model = SimpleModel().cuda()
model.eval()  # 错误发生在评估模式下调用 backward()

# 模拟训练步骤
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x = torch.randn(1, 10).cuda()
y = model(x)

# 尝试反向传播（在 eval 模式下会报错）
loss = y.mean()
loss.backward()  # 这里会触发 "RuntimeError: backward_input can only be called in training mode"