import torch
import yaml
import numpy as np
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from models.wrapper import FlexoraWrapper
from search.engine import BilevelSearchEngine
from data.loader import get_dataloaders
from huggingface_hub import login
login('hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx')

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running search on {device}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    peft_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['lora_alpha'],
        target_modules=cfg['lora']['target_modules'],
        lora_dropout=cfg['lora']['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.print_trainable_parameters()

    flexora_model = FlexoraWrapper(peft_model, num_layers=cfg['model']['num_layers'])

    train_loader, val_loader = get_dataloaders(
        cfg['model']['name'], 
        cfg['search']['batch_size'],
        cfg['search']['max_length']
    )

    engine = BilevelSearchEngine(
        flexora_model, 
        lr_model=float(cfg['search']['lr_model']),
        lr_alpha=float(cfg['search']['lr_alpha'])
    )

    print("Starting Flexible Layer Selection...")
    val_iter = iter(val_loader)
    
    for step, batch_train in enumerate(train_loader):
        if step >= cfg['search']['search_steps']:
            break
            
        try:
            batch_val = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch_val = next(val_iter)
        
        batch_train = {k: v.to(device) for k, v in batch_train.items()}
        batch_val = {k: v.to(device) for k, v in batch_val.items()}
        
        loss_t, loss_v = engine.step(batch_train, batch_val)
        
        if step % 10 == 0:
            print(f"Step {step}: Train Loss={loss_t:.4f}, Val Loss={loss_v:.4f}")

    # Select Layers (Positive Alpha Strategy)
    alphas = flexora_model.get_layer_importance()
    print("\nFinal Alphas:", alphas)
    
    # Proposition 1: Select layers where alpha > average (or > 0 if unnormalized)
    threshold = np.mean(alphas)
    selected_indices = np.where(alphas > threshold)[0]
    
    print(f"Selected {len(selected_indices)} layers: {selected_indices}")
    np.save("selected_layers.npy", selected_indices)

if __name__ == "__main__":
    main()
