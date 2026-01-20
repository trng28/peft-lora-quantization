import torch
from torch.optim import AdamW

class BilevelSearchEngine:
    def __init__(self, flexora_wrapper, lr_model=1e-4, lr_alpha=3e-2):
        self.wrapper = flexora_wrapper

        self.model_params = [
            p for n, p in self.wrapper.peft_model.named_parameters() 
            if p.requires_grad
        ]

        self.optimizer_theta = AdamW(self.model_params, lr=lr_model)
        self.optimizer_alpha = AdamW([self.wrapper.alpha_params], lr=lr_alpha)

    def step(self, batch_train, batch_val):
        """
        Algorithm 1:
        1. Update Theta on Train
        2. Update Alpha on Val (using updated Theta)
        """
        
        # --- Step A: Update Theta (LoRA weights) on Train Set ---
        self.optimizer_theta.zero_grad()
        outputs_train = self.wrapper(**batch_train)
        loss_train = outputs_train.loss
        loss_train.backward()
        self.optimizer_theta.step()
        
        # --- Step B: Update Alpha (Layer selection) on Val Set ---
        # Note: In strict bilevel optim (DARTS), use a virtual step.
        # Use the alternating approximation (First-order) to save VRAM.
        
        self.optimizer_alpha.zero_grad()
        outputs_val = self.wrapper(**batch_val)
        loss_val = outputs_val.loss
        
        # Compute gradients w.r.t alpha
        loss_val.backward()
        self.optimizer_alpha.step()
        
        return loss_train.item(), loss_val.item()
