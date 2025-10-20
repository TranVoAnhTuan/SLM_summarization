# trainer_module.py
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
import os

def train_model(model, tokenized, tokenizer, output_dir="./t5small_lora"):
    """
    Train LoRA/QLoRA T5 model with subset (chunked) training.
    Áp dụng: gradient checkpointing, fp16, dynamic padding, early stopping, resume checkpoint.
    """

    # 🟩 Cấu hình huấn luyện
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,          # nhỏ vì GPU 4GB
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        fp16=True,                              # mixed precision (tự động FP16)
        save_strategy="epoch",                  # lưu checkpoint mỗi epoch
        eval_strategy="epoch",            # 🔹 tên đúng, tránh lỗi
        load_best_model_at_end=True,            # chọn model tốt nhất
        metric_for_best_model="eval_loss",
        save_total_limit=3,
        logging_steps=100,
        report_to="none",
        remove_unused_columns=False,            # cần cho seq2seq training
    )

    # 🟩 Dynamic padding (giảm VRAM)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8
    )

    # 🟩 Tạo Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=None,  # gán sau
        eval_dataset=tokenized["validation"].select(range(1000)),
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 🟩 Chia nhỏ tập train để tránh đầy VRAM
    train_data = tokenized["train"]
    chunk_size = 50000  # bạn có thể giảm xuống 20000 nếu VRAM thấp
    num_chunks = len(train_data) // chunk_size + (1 if len(train_data) % chunk_size != 0 else 0)

    print(f"Total train samples: {len(train_data)}")
    print(f"Training in {num_chunks} chunks of {chunk_size} samples each")

    # 🟩 Train từng phần (và resume nếu crash)
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(train_data))
        subset = train_data.select(range(start, end))
        print(f"\n Training chunk {i + 1}/{num_chunks} — samples {start} to {end}")

        trainer.train_dataset = subset

        # Resume nếu checkpoint tồn tại
        ckpt_path = f"{output_dir}/chunk_{i}"
        if os.path.exists(ckpt_path):
            print(f" Resuming from checkpoint model weights only: {ckpt_path}")
            model = model.from_pretrained(ckpt_path)
            trainer.model = model
        else:
            print(" No checkpoint found, starting fresh for this chunk.")

        trainer.train()


        # Lưu checkpoint sau mỗi chunk
        save_path = f"{output_dir}/chunk_{i + 1}"
        os.makedirs(save_path, exist_ok=True)
        trainer.save_model(save_path)
        trainer.save_state()
        print(f"Saved checkpoint to {save_path}")

    print("Training completed across all chunks!")
    return trainer.model
