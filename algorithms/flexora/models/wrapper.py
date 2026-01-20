import torch
import torch.nn as nn
from peft import PeftModel
from utils.helpers import get_layer_index_from_name

class FlexoraWrapper(nn.Module):
    def __init__(self, peft_model: PeftModel, num_layers: int):

        super().__init__()
        self.peft_model = peft_model
        self.num_layers = num_layers
        self.alpha_params = nn.Parameter(torch.zeros(num_layers))
        self.hooks = []
        self._register_flexora_hooks()

    def get_normalized_alphas(self):
        """
        Equation 4: 
        alpha_hat = exp(alpha) / sum(exp(alpha)) * N
        """
        softmax_val = torch.nn.functional.softmax(self.alpha_params, dim=0)
        return softmax_val * self.num_layers

    def _register_flexora_hooks(self):
        """
        h = Wx + alpha * BAx
        """
        
        def hook_fn(layer_idx):

            def forward_hook(module, input, output):
                alphas = self.get_normalized_alphas()
                scale = alphas[layer_idx]
                return output * scale
            return forward_hook

        for name, module in self.peft_model.named_modules():
            if "lora_" in name and "default" in name: 
                pass
            
            if isinstance(module, nn.Linear) and (("lora" in name) or hasattr(module, "lora_A")):
                idx = get_layer_index_from_name(name)
                if idx is not None and idx < self.num_layers:
                    h = module.register_forward_hook(hook_fn(idx))
                    self.hooks.append(h)

    def forward(self, **kwargs):
        return self.peft_model(**kwargs)

    def get_layer_importance(self):
        return self.get_normalized_alphas().detach().cpu().numpy()
