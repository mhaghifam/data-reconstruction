import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d, n_classes, h1=1500, h2=1500):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_classes)
        )

    def forward(self, x):
        return self.net(x)