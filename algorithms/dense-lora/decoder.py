import torch 
import torch.nn as nn

class DenseLoRADecoder(nn.Module):

    def __init__(self, out_features, rank, activation):
        super().__init__()
        self.linear = nn.Linear(rank, out_features, bias = False)
        self.activation = activation()

        nn.init.zeros_(self.linear.weight)

    def forward(self, h):
        return self.activation(self.linear(h))