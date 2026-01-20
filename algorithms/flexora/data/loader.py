from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

def get_dataloaders(model_id, batch_size, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")
    
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = split_dataset["train"]
    val_ds = split_dataset["test"]

    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )

    tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_val = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=True, collate_fn=collator)

    return train_loader, val_loader
