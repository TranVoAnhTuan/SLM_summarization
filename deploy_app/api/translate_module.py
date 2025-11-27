from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class TranslateModule:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu") -> None:
        print("Loading translation module")
        self.device = device
        self.en2vi_tokenizer = AutoTokenizer.from_pretrained("./models/translate/opus-mt-en-vi", local_files_only=True)
        self.en2vi_model = AutoModelForSeq2SeqLM.from_pretrained("./models/translate/opus-mt-en-vi", local_files_only=True).to(device)

        self.vi2en_tokenizer = AutoTokenizer.from_pretrained("./models/translate/opus-mt-vi-en", local_files_only=True)
        self.vi2en_model = AutoModelForSeq2SeqLM.from_pretrained("./models/translate/opus-mt-vi-en", local_files_only=True).to(device)

        print("Tranlation modules loaded successfully")

    def en_to_vi(self, text: str, max_tokens: int = 800) -> str:
        inputs = self.en2vi_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.en2vi_model.generate(**inputs, max_new_tokens=max_tokens)
        translated = self.en2vi_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated.strip()
    
    def vi_to_en(self, text: str, max_tokens: int = 256) -> str:
        inputs = self.vi2en_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.vi2en_model.generate(**inputs, max_new_tokens=max_tokens)
        translated = self.vi2en_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated.strip()
    
    def translate_pipeline(self, text: str, target_lang: str = "vi") -> str:
        if target_lang.lower() == "vi":
            return self.en_to_vi(text)
        elif target_lang.lower() == "en":
            return self.vi_to_en(text)
        else:
            raise ValueError("target_lang must be 'vi' or 'en'.")