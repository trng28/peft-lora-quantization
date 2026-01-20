import torch
import torch.nn as nn

class DenseLoRAEncoder(nn.Module):

    def __init__(self, in_features, rank, activation):
        super().__init__()
        self.linear = nn.Linear(in_features, rank, bias = False)
        self.activation = activation()

        nn.init.kaiming_uniform_(self.linear.weight, a = 5 ** 0.5)

    def forward(self, h):
        return self.activation(self.linear(h))
    
