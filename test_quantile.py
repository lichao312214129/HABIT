import torch
x = torch.tensor([[1., 2., float("nan")], [float("nan"), 3., 4.]])
print(torch.nanquantile(x, 0.1, dim=1))
