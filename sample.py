import torch

print(torch.__version__)

a = torch.rand((100, 2000)).cuda()
b = torch.rand((100, 2000)).cuda()

c = torch.mul(a, b)