from transformers import AutoTokenizer
import os
from datasets import load_from_disk

def tokenize_datasets(
    processed,
    model_name="./models/t5-small",
    max_input=800,
    max_target=256,
    num_proc=4,
    save_dir="./tokenized_cnn_dm"
):
    if os.path.exists(save_dir):
        print(f"Found cached tokenized data at {save_dir}")
        print(f"Loading from disk...")
    
    try:
        tokenized = load_from_disk(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        
        print(f"Loaded tokenized dataset from cache")
        print(f"   Train: {len(tokenized['train']):,} samples")
        print(f"   Validation: {len(tokenized['validation']):,} samples")
        
        return tokenized, tokenizer
        
    except Exception as e:
        print(f"  Failed to load cache: {e}")
        print(f"   Re-tokenizing from scratch...")

    print(f"Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_input,
        )

        labels = tokenizer(
            batch["summary"],
            truncation=True,
            padding="max_length",
            max_length=max_target,
            )
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    print("Tokenizing dataset... (parallelized)")
    tokenized = processed.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc, 
        remove_columns=processed["train"].column_names,
    )

    os.makedirs(save_dir, exist_ok=True)
    tokenized.save_to_disk(save_dir)
    print(f"Tokenized dataset saved to {save_dir}")

    return tokenized, tokenizer
