from django.shortcuts import render
from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import json
from .translate_module import TranslateModule

# --- Load mô hình khi khởi động ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

BASE_MODEL = "./models/t5-small"
FINETUNED_DIR = "./t5small_lora/chunk_29"

print("Loading tokenizer & quantized model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, device_map="auto", local_files_only=True
)
model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
model.eval()
print("Model loaded successfully.")

# Module dịch Anh ↔ Việt
translator = TranslateModule(device="cuda" if torch.cuda.is_available() else "cpu")

# --- Trang giao diện chính ---
def home(request):
    return render(request, "api/summarize.html")


# --- API tóm tắt văn bản ---
def summarize_text(request):
    if request.method != "POST":
        return JsonResponse({"error": "Use POST method"}, status=400)

    data = json.loads(request.body.decode("utf-8"))
    text = data.get("text", "")
    lang = data.get("lang", "vi")

    if not text.strip():
        return JsonResponse({"error": "Empty text input"}, status=400)

    try:
        print("Nhận input:", text[:150].replace("\n", " "), "...")
        device = model.device

        # --- B1: Nếu người dùng nhập tiếng Việt thì dịch sang tiếng Anh ---
        if lang == "vi":
            text = translator.vi_to_en(text)
            print("Dịch sang tiếng Anh thành công.")

        # --- B2: Tạo prompt và sinh tóm tắt ---
        prompt = f"Summarize the following text in less than four sentences:\n\n{text}"

        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800).to(device)
            outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4, temperature= 0.9)
            summary_en = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("Sinh tóm tắt tiếng Anh xong.")

        # --- B3: Nếu người dùng chọn tiếng Việt → dịch kết quả ngược lại ---
        if lang == "vi":
            summary = translator.en_to_vi(summary_en)
            print("Dịch ngược sang tiếng Việt thành công.")
        else:
            summary = summary_en

        return JsonResponse({"summary": summary.strip()})

    except Exception as e:
        print("Lỗi trong quá trình generate:", e)
        return JsonResponse({"error": str(e)}, status=500)
