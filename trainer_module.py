import os
import json
import csv
import time
import gc
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from config import Config

class TrainerModule:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.chunk_size = cfg.chunk_size

    def train(self, model, tokenized, tokenizer):
        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            num_train_epochs=self.cfg.num_train_epochs,
            fp16=True,
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
        num_chunks = len(train_data) // self.chunk_size + (1 if len(train_data) % self.chunk_size else 0)

        print(f"Total train samples: {len(train_data)}")
        print(f"Training in {num_chunks} chunks of {self.chunk_size} samples each")

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, len(train_data))
            subset = train_data.select(range(start, end))
            trainer.train_dataset = subset

            ckpt_path = f"{self.output_dir}/chunk_{i+1}"
            print(f"\nTraining chunk {i+1}/{num_chunks} — samples {start} to {end}")
            start_time = time.time()

            if os.path.exists(ckpt_path):
                print(f"Resuming from checkpoint: {ckpt_path}")
                result = trainer.train(resume_from_checkpoint=ckpt_path)
            else:
                result = trainer.train()

            elapsed = round(time.time() - start_time, 2)
            os.makedirs(ckpt_path, exist_ok=True)
            trainer.save_model(ckpt_path)
            trainer.state.save_to_json(os.path.join(ckpt_path, "trainer_state.json"))
            print(f"Saved checkpoint to {ckpt_path}")

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

            with open(os.path.join(ckpt_path, "training_log.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            log_csv = os.path.join(self.output_dir, "metrics.csv")
            write_header = not os.path.exists(log_csv)
            with open(log_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(metrics)

            print(f"Metrics logged to {log_csv}")

        print("Training completed across all chunks!")
        return trainer.model
