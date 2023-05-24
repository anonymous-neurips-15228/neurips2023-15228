import torch.nn as nn

class NLayerLeakyMLP(nn.Module):

    def __init__(self, in_features, out_features, num_layers, hidden_dim):
        super().__init__()
        layers = [ ]
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(in_features, hidden_dim))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)