from evaluate import load

def evaluate_model(model, tokenizer, dataset):
    rouge = load("rouge")

    def generate_summary(batch):
        inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=256)
        batch["predicted"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return batch

    preds = dataset.select(range(200)).map(generate_summary, batched=True, batch_size=2)
    results = rouge.compute(predictions=preds["predicted"], references=preds["summary"])
    return results
