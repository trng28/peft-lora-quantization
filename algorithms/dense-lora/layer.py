import torch 
import torch.nn as nn
from .encoder import DenseLoRAEncoder
from .decoder import DenseLoRADecoder
from .adapter import DenseLoRAAdapter

class DenseLoRALayer(nn.Module):

    def __init__(self, base_layer, encoder, decoder, rank, scaling):

        super().__init__()

        self.base = base_layer
        self.base.weight.requires_grad_(False)

        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.encoder = encoder
        self.decoder = decoder
        self.adapter = DenseLoRAAdapter(rank)

        self.scaling = scaling

    def forward(self, x):
        # x: (batch, *, in_features)

        base_out = self.base(x)  

        h -= self.encoder(x)
        h = self.adapter(h)
        delta = self.decoder(h)

        return base_out + self.scaling * delta