import torch.nn as nn

class ReLUNetwork(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, depth=4, width=100):
        super().__init__()

        layers = [nn.Linear(input_dim, width), nn.ReLU()]

        for _ in range(depth - 2):
            layers += [nn.Linear(width, width), nn.ReLU()]

        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
