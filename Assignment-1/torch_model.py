import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(500, 250, 100), output_dim=10, activation="relu"):
        super(MLP, self).__init__()
        
        layers = []
        sizes = [input_dim] + list(hidden_dims)
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
        
        layers.append(nn.Linear(sizes[-1], output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)