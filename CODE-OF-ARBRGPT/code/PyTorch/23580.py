import torch
import timeit

a = torch.randn(1000, 10)

# 多次运行取平均时间
time = timeit.timeit(lambda: torch.qr(a), number=10)
print(f"torch.qr time: {time/10:.4f} ")