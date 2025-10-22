import torch
import torch.nn as nn

x = torch.rand(4, 3, 8, 8)

conv = nn.Conv2d(3, 6, 3)
fc = nn.Linear(216, 10)

x = conv(x).flatten(start_dim=1)

output = fc(x)
