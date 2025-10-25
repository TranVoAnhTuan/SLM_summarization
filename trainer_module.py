import os
import json
import csv
import time
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
import torch, gc

def train_model(model, tokenized, tokenizer, output_dir="./t5small_lora"):
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        num_train_epochs=2,
        fp16=True,
        # gradient_checkpointing=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        logging_steps=50,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=None,
        eval_dataset=tokenized["validation"].select(range(200)),
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    train_data = tokenized["train"]
    chunk_size = 10000  
    num_chunks = len(train_data) // chunk_size + (1 if len(train_data) % chunk_size else 0)
    # num_chunks = 14

    print(f"Total train samples: {len(train_data)}")
    print(f"Training in {num_chunks} chunks of {chunk_size} samples each")

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(train_data))
        subset = train_data.select(range(start, end))
        trainer.train_dataset = subset

        ckpt_path = f"{output_dir}/chunk_{i+1}"
        print(f"\n Training chunk {i+1}/{num_chunks} — samples {start} to {end}")
        start_time = time.time()

        if os.path.exists(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            result = trainer.train(resume_from_checkpoint=ckpt_path)
        else:
            result = trainer.train()

        elapsed = round(time.time() - start_time, 2)

        save_path = f"{output_dir}/chunk_{i+1}"
        os.makedirs(save_path, exist_ok=True)
        trainer.save_model(save_path)
        trainer.state.save_to_json(os.path.join(save_path, "trainer_state.json"))
        print(f"Saved checkpoint to {save_path}")
        torch.cuda.empty_cache()
        gc.collect()

        metrics = result.metrics
        metrics.update({
            "chunk": i + 1,
            "samples": f"{start}-{end}",
            "time_sec": elapsed,
            "train_batch_size": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
        })

        with open(os.path.join(save_path, "training_log.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        log_csv = os.path.join(output_dir, "metrics.csv")
        write_header = not os.path.exists(log_csv)
        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

        print(f"Metrics logged to {log_csv}")

    print("Training completed across all chunks!")
    return trainer.model
