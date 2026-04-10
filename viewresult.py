import torch

weights = torch.load("outputs/deeprawnet.pth", map_location="cpu")
for key, value in weights.items():
    print(key, value.shape)