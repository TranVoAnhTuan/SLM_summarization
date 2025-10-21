from data_loader import load_and_preprocess
from tokenizer_module import tokenize_datasets
from model_module import load_model
from trainer_module import train_model
from evaluate_module import evaluate_model
from inference_module import summarize_text

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    processed = load_and_preprocess()

    print("Tokenizing dataset...")
    tokenized, tokenizer = tokenize_datasets(processed)

    print("⚙️ Loading model (local T5-small + LoRA)...")
    model = load_model()

    try:
        print("Starting training...")
        model = train_model(model, tokenized, tokenizer)
    except RuntimeError as e:
        print("GPU crashed, saving log...")
        with open("error_log.txt", "a") as f:
            f.write(str(e) + "\n")

    print("Evaluating model...")
    evaluate_model(model, tokenizer, processed)  # ✅ dùng processed, không dùng tokenized

    print("Example inference:")
    text = """The President of the United States announced new climate policies aiming to reduce emissions by 40%."""
    summary = summarize_text(model, tokenizer, text)
    print(summary)
