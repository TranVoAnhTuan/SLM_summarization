import os
from datasets import load_from_disk
from transformers import AutoTokenizer
from config import Config

class TokenizerModule:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.max_input = cfg.max_input
        self.max_target = cfg.max_target
        self.num_proc = cfg.num_proc
        self.save_dir = cfg.tokenized_cache

    def tokenize(self, processed):
        if os.path.exists(self.save_dir):
            print(f"Found cached tokenized data at {self.save_dir}")
            try:
                tokenized = load_from_disk(self.save_dir)
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                print(f"Loaded tokenized dataset from cache")
                print(f"   Train: {len(tokenized['train']):,} samples")
                print(f"   Validation: {len(tokenized['validation']):,} samples")
                return tokenized, tokenizer
            except Exception as e:
                print(f"Failed to load cache: {e}")
                print("Re-tokenizing from scratch...")

        print(f"Loading tokenizer from {self.model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)

        def tokenize_fn(batch):
            model_inputs = tokenizer(
                batch["prompt"],
                truncation=True,
                padding="max_length",
                max_length=self.max_input,
            )
            labels = tokenizer(
                batch["summary"],
                truncation=True,
                padding="max_length",
                max_length=self.max_target,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        print("Tokenizing dataset... (parallelized)")
        tokenized = processed.map(
            tokenize_fn,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=processed["train"].column_names,
        )

        os.makedirs(self.save_dir, exist_ok=True)
        tokenized.save_to_disk(self.save_dir)
        print(f"Tokenized dataset saved to {self.save_dir}")

        return tokenized, tokenizer
