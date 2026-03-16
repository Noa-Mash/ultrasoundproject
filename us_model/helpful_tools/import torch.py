import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(1000, 1000, device=device)
y = torch.matmul(x, x)

print(y.device)
