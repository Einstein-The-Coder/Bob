import torch

x = torch.load("chess_model_weights.pth", map_location="cpu",weights_only = False)

print(type(x))

if isinstance(x, dict):
    print(x.keys())