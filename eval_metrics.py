import os
import json
from config import Config
from data_loader import DataLoader
from evaluate_module import Evaluator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

cfg = Config()
cfg.num_eval_samples = None     

data_loader = DataLoader(cfg)
dataset = data_loader.load_and_preprocess()
evaluator = Evaluator(cfg)

reuse_cache = True

models = {
    "T5-small": {
        "path": "./t5small_lora/chunk_29",
        "model_name": "./models/t5-small"
    },
    "T5-base": {
        "path": "./t5base_model/t5base_lora/chunk_20",
        "model_name": "./models/t5-base"
    },
}

for name, info in models.items():
    print(f"\n Evaluating {name} (METRICS ONLY) ...")

    cache_path = f"./evaluation_logs/preds_{name}.jsonl"

    # Load model + tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(info["path"], device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(info["model_name"])

    metrics = evaluator.evaluate(
        model=model,
        tokenizer=tokenizer,
        dataset={"test": dataset["test"]},
        cache_path=cache_path,
        reuse_cache=reuse_cache
    )

    print(f"\n {name} metrics:")
    for k, v in metrics.items():
        print(f"{k:<20}: {v}")
