import torch

weights = torch.load("outputs/deeprawnet_model.pth", map_location="cpu")
for key, value in weights.items():
    print(key, value.shape)