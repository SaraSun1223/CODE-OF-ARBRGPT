import os

import torch

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp





class Model(torch.nn.Module):



    def __init__(self):

        super().__init__()

        self.layer = torch.nn.Linear(8, 16)

        self.token = torch.nn.Parameter(torch.zeros(8))



    def forward(self, x):

        x[[True, False]] = self.token

        x = self.layer(x)

        return x



def main(rank, world_size):



    os.environ['MASTER_ADDR'] = 'localhost'

    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)



    model = Model().to(rank)

    ddp_model = DDP(model, device_ids=[rank])



    x = torch.randn(2, 8, device=rank)

    y = ddp_model(x).mean()

    y.backward()





    torch.cuda.synchronize()

    dist.barrier()

    dist.destroy_process_group()



if __name__ == "__main__":

    mp.spawn(main, args=(2,), nprocs=2, join=True)