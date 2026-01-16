import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d, n_classes, h1=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h1),
            nn.Sigmoid(),
            nn.Linear(h1, n_classes)
        )

    def forward(self, x):
        return self.net(x)