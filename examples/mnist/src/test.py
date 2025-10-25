import torch
import torch.nn as nn

# x = torch.rand(4, 3, 8, 8)
#
# conv = nn.Conv2d(3, 6, 3)
# fc = nn.Linear(216, 10)
#
# x = conv(x).flatten(start_dim=1)
#
# output = fc(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l = nn.Linear(10, 20)  # Linear<10, 20>
        self.r = nn.ReLU()          # Relu
        self.l2 = nn.Linear(20, 2)  # Linear<20, 2>

    def forward(self, x):
        x = self.l(x)
        x = self.r(x)
        x = self.l2(x)
        return x


# Equivalent to VarMap / Vs setup (not needed in PyTorch)
model = Model()

# Create an input tensor equivalent to Tensor<Rank2<2, 10>> = Tensor::ones()
xs = torch.ones((2, 10))

# Forward pass
ys = model(xs)

print(ys)
