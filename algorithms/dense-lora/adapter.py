import torch 
import torch.nn as nn



class DenseLoRAAdapter(nn.Module):

    def __init__(self, rank):

        super().__init__()
        self.M = nn.Parameter(torch.empty(rank,rank))

        nn.init.kaiming_uniform_(self.M, a = 5 ** 0.5)

    def forward(self, h):
        return h @ self.M.T