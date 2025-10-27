import os
import json
import csv
from datetime import datetime
import torch
from tqdm import tqdm
import evaluate
from config import Config

class Evaluator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.save_dir = cfg.eval_logs
        os.makedirs(self.save_dir, exist_ok=True)

    def _generate_summary(self, sample, model, tokenizer):
        with torch.no_grad(), torch.amp.autocast("cuda"):
            inputs = tokenizer(
                sample["prompt"],
                truncation=True,
                padding="max_length",
                max_length=self.cfg.max_input,
                return_tensors="pt"
            ).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=self.cfg.max_target, num_beams=4)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def evaluate(self, model, tokenizer, dataset):
        num_samples = self.cfg.num_eval_samples
        sample_dataset = dataset["validation"].select(range(num_samples))

        print(f"Evaluating model on {num_samples} samples...")

        preds, refs = [], []
        for sample in tqdm(sample_dataset, total=num_samples):
            pred = self._generate_summary(sample, model, tokenizer)
            preds.append(pred)
            refs.append(sample["summary"])

        rouge = evaluate.load("rouge")
        bleu = evaluate.load("sacrebleu")
        meteor = evaluate.load("meteor")
        bertscore = evaluate.load("bertscore")

        rouge_scores = rouge.compute(predictions=preds, references=refs)
        metrics = {
            "ROUGE-1": rouge_scores["rouge1"],
            "ROUGE-2": rouge_scores["rouge2"],
            "ROUGE-L": rouge_scores["rougeL"],
            "BLEU": bleu.compute(predictions=preds, references=[[r] for r in refs])["score"]/100.0,
            "METEOR": meteor.compute(predictions=preds, references=refs)["meteor"],
        }
        bert = bertscore.compute(predictions=preds, references=refs, lang="en")
        metrics["BERTScore (F1)"] = sum(bert["f1"]) / len(bert["f1"])

        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"{k:<15}: {v:.4f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(self.save_dir, f"eval_{timestamp}.json")
        csv_path = os.path.join(self.save_dir, "eval_metrics.csv")

        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

        print(f"\nMetrics saved to {json_path}")
        print(f"Appended metrics to {csv_path}")
        return metrics
