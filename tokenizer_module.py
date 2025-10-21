from transformers import AutoTokenizer
import os

def tokenize_datasets(
    processed,
    model_name="./models/t5-small",
    max_input=800,
    max_target=256,
    num_proc=4,
    save_dir="./tokenized_cnn_dm"
):
    """
    Tokenize dataset for T5 summarization task.
    - processed: DatasetDict (đã có 'prompt' và 'summary')
    - model_name: local path hoặc HuggingFace model ID
    - max_input: giới hạn độ dài input tokens
    - max_target: giới hạn độ dài summary tokens
    - num_proc: số tiến trình chạy song song
    - save_dir: nơi lưu dataset đã tokenized
    """
    print(f"Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    def tokenize_fn(batch):
        # tokenize input (article/prompt)
        model_inputs = tokenizer(
            batch["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_input,
        )

        # tokenize labels (summary)
        # with tokenizer.as_target_tokenizer():
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
        num_proc=num_proc,  # 🔹 chạy song song
        remove_columns=processed["train"].column_names,
    )

    os.makedirs(save_dir, exist_ok=True)
    tokenized.save_to_disk(save_dir)
    print(f"Tokenized dataset saved to {save_dir}")

    return tokenized, tokenizer
