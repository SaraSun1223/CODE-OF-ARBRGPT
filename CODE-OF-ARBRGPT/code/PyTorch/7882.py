import torch
import random

from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __len__(self):
        return 10000
    def __getitem__(self, index):
        print(index, torch.rand(2, 2).sum().item(), random.uniform(0, 1))
        return 1

seed = 2018
random.seed(seed)
torch.manual_seed(seed)
loader = DataLoader(Data(), num_workers=4, shuffle=True)

for x in loader:
    print('-'*10)
    break