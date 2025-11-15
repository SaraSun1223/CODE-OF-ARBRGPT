import torch

module = torch.nn.Sequential(

        torch.nn.Linear(20, 100),

        torch.nn.BatchNorm1d(100)

).cuda()

print(set([p.device for p in module.parameters()]))

sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)

print(set([p.device for p in sync_bn_module.parameters()]))