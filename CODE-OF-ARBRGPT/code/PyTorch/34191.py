import torch



a = torch.ones(size=(5, ), device='cuda', requires_grad=True).cuda()

out, _ = torch.topk(a, 2)

out.mean().backward()

print('step 1 backward with any value.')



a = torch.full(size=(5, ), fill_value=float('nan'),

               device='cuda', requires_grad=True).cuda()

out, _ = torch.topk(a, 2)

out.mean().backward()

print('step 2 backward with NAN.')


