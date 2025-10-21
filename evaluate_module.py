# evaluate_module.py
import os
import json
import csv
from datetime import datetime
import torch
from tqdm import tqdm
import evaluate

from transformers import T5ForConditionalGeneration, AutoTokenizer



def generate_summary(batch, model, tokenizer, max_new_tokens=256):
    with torch.no_grad(), torch.amp.autocast("cuda"):
        inputs = tokenizer(
            batch["prompt"],
            truncation=True,
            padding="max_length",
            max_length=800,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4
        )

    summary = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return summary



def evaluate_model(model, tokenizer, dataset, num_samples=50, save_dir="./evaluation_logs"):
    os.makedirs(save_dir, exist_ok=True)

    sample_dataset = dataset["validation"].select(range(num_samples))

    print(f"🔍 Evaluating model on {num_samples} samples...")

    preds, refs = [], []
    for sample in tqdm(sample_dataset, total=num_samples):
        pred = generate_summary(sample, model, tokenizer)[0]
        preds.append(pred) 
        refs.append(sample["summary"])

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    # trong hàm evaluate_model
    rouge_scores = rouge.compute(predictions=preds, references=refs)

    metrics = {
        "ROUGE-1": rouge_scores["rouge1"],
        "ROUGE-2": rouge_scores["rouge2"],
        "ROUGE-L": rouge_scores["rougeL"],
        "BLEU": bleu.compute(predictions=preds, references=[[r] for r in refs])["score"],
        "METEOR": meteor.compute(predictions=preds, references=refs)["meteor"],
    }

    bert = bertscore.compute(predictions=preds, references=refs, lang="en")
    metrics["BERTScore (F1)"] = sum(bert["f1"]) / len(bert["f1"])

    # --- In kết quả ---
    print("\n Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k:<15}: {v:.4f}")

    # --- Lưu kết quả ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(save_dir, f"eval_{timestamp}.json")
    csv_path = os.path.join(save_dir, "eval_metrics.csv")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)

    print(f"\n Metrics saved to {json_path}")
    print(f" Appended metrics to {csv_path}")

    return metrics



if __name__ == "__main__":
    from datasets import load_dataset

    model_path = "./t5small_lora/chunk_14"
    print(f"Loading model from {model_path} ...")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    results = evaluate_model(model, tokenizer, dataset, num_samples=50)
