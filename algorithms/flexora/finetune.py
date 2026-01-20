import torch
import yaml
import numpy as np
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from utils.helpers import get_specific_target_modules 
from data.loader import get_dataloaders

def main():
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    try:
        selected_indices = np.load("selected_layers.npy")
    except FileNotFoundError:
        print("Error: Run search step first!")
        return

    print(f"Fine-tuning on {len(selected_indices)} layers: {selected_indices}")

    # Setup Specific Target Modules
    target_modules = get_specific_target_modules(
        selected_indices, 
        cfg['lora']['target_modules']
    )
    
    print(f"Example targets: {target_modules[:3]}")

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['name'],
        dtype=torch.bfloat16,  
        device_map="auto"
    )

    peft_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['lora_alpha'],
        target_modules=target_modules, 
        lora_dropout=cfg['lora']['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, peft_config)
    
    # verify Flexora 
    model.print_trainable_parameters()
    
    train_loader, _ = get_dataloaders(
        cfg['model']['name'], 
        cfg['finetune']['batch_size'],
        cfg['search']['max_length']
    )
    
    args = TrainingArguments(
        output_dir="./flexora_output",
        num_train_epochs=cfg['finetune']['epochs'],
        learning_rate=float(cfg['finetune']['lr']),
        per_device_train_batch_size=cfg['finetune']['batch_size'],
        logging_steps=10,
        bf16=True,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_loader.dataset, 
    )
    
    print(">>> Starting Fine-Tuning Stage...")
    trainer.train()

if __name__ == "__main__":
    main()
