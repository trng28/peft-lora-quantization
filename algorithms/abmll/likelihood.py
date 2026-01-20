import torch
import torch.nn as nn
import torch.nn.functional as F

class RegressionLikelihood(nn.Module):

    def __init__(self, phi_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(phi_dim, out_dim)

    def forward(self, x, phi):
        return self.linear(phi).unsequeeze(0)
    

def log_likelihood(model, D, phi):
    x, y = D
    y_pred = model(x, phi)
    return -F.mse_loss(y_pred, y, reduction='mean')