import torch
import timeit

torch.set_num_threads(1)  # 确保单线程执行
x = torch.empty(1024*1024*100, dtype=torch.uint8)
y = torch.empty(1024*1024*100, dtype=torch.float)

# 预热
y.copy_(x)

# 多次运行取平均时间
time = timeit.timeit(lambda: y.copy_(x), number=10)
print(f"y.copy_(x) 运行时间: {time/10:.4f} 秒/次")
