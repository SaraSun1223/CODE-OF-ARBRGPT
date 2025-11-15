import torch



# The following ones will fail.

a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int64).pow(4); print(a)

a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int32).pow(4); print(a)

a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int16).pow(4); print(a)

a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int8).pow(4); print(a)



# The following ones will be okay.

a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int64).pow(2); print(a)

a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int32).pow(2); print(a)

a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int16).pow(2); print(a)

a = torch.tensor([-1,  1], device='cuda:0', dtype=torch.int8).pow(2); print(a)