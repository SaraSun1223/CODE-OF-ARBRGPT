import torch

import time



from torch.utils.data import TensorDataset, DataLoader



dataset = TensorDataset(torch.randn(10240, 2))

loader = DataLoader(dataset, batch_size=128, num_workers=2, pin_memory=True, drop_last=False)



for epoch in range(10):

    for idx, data in enumerate(loader):

        data = data[0].cuda()

        if idx == 10240/128-1:

            ts = time.time()

    print("Exit epoch {} elapsed {:.2f}s".format(epoch, time.time()-ts))