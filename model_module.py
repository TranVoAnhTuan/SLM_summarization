from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

def load_model(model_name="./models/t5-small"):
    print(f"Loading local model from {model_name} ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q", "k", "v", "o", "wi", "wo"],
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)
    torch.cuda.empty_cache()
    print("Model loaded with 4-bit quantization + LoRA")
    return model
