import torch 
import torch.nn as nn
from torch.distributions import Normal

class InferenceNet(nn.Module):

    def __init__(self, input_dim, phi_dim, hidden_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, phi_dim)
        self.logvar = nn.LLinear(hidden_dim, phi_dim)

    def forward(self, D):
        h = self.encoder(D.mean(dim=0))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return Normal(mu, torch.exp(0.5 * logvar))
    
