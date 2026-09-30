import torch.nn as nn

ACTIVATIONS = {'sigmoid': nn.Sigmoid, 'relu': nn.ReLU}


class MLP(nn.Module):
    def __init__(self, d, n_classes, h1=500, activation='sigmoid', center_input=False):
        super().__init__()
        self.center_input = center_input
        self.net = nn.Sequential(
            nn.Linear(d, h1),
            ACTIVATIONS[activation](),
            nn.Linear(h1, n_classes)
        )

    def forward(self, x):
        if self.center_input:
            # {0,1} -> {-1,+1}
            x = 2 * x - 1
        return self.net(x)
