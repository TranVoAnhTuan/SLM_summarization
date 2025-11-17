import time, os, csv, random
from config import Config
from data_loader import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

try:
    import psutil
    HAS_PSUTIL = True
except:
    HAS_PSUTIL = False

def bytes_to_mb(x): 
    return round(x / (1024**2), 2)

# Load config + dataset
cfg = Config()
data_loader = DataLoader(cfg)
dataset = data_loader.load_and_preprocess()

sample = random.choice(dataset["test"])
prompt = sample["prompt"]

models = {
    "T5-small": {"path": "./t5small_lora/chunk_29", "model_name": "./models/t5-small"},
    "T5-base":  {"path": "./t5base_model/t5base_lora/chunk_20", "model_name": "./models/t5-base"},
}

save_csv = "./evaluation_logs/performance_single_sample.csv"
os.makedirs("./evaluation_logs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

for name, info in models.items():
    print(f"\nMeasuring performance of {name} on ONE TEST SAMPLE ...")

    # Clear VRAM
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(info["path"], device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(info["model_name"])

    # VRAM after load
    vram_after_load = (
        bytes_to_mb(torch.cuda.memory_allocated())
        if device.type == "cuda"
        else None
    )

    # Reset peak tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Prepare input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_input
    ).to(model.device)

    # Measure inference time
    t0 = time.time()
    _ = model.generate(**inputs, max_new_tokens=cfg.max_target, num_beams=4)
    inference_time = time.time() - t0

    # Peak VRAM during inference
    peak_vram_infer = (
        bytes_to_mb(torch.cuda.max_memory_allocated())
        if device.type == "cuda"
        else None
    )

    # RAM usage
    ram_mb = (
        bytes_to_mb(psutil.Process(os.getpid()).memory_info().rss)
        if HAS_PSUTIL
        else None
    )

    # Store results
    row = {
        "model": name,
        "inference_time(s)": round(inference_time, 4),
        "vram_after_load_MB": vram_after_load,
        "peak_vram_infer_MB": peak_vram_infer,
        "ram_MB": ram_mb,
    }

    results.append(row)

write_header = not os.path.exists(save_csv)
with open(save_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    if write_header:
        writer.writeheader()
    writer.writerows(results)

print(f"\nPerformance saved to: {save_csv}")