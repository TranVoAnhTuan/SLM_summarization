from config import Config

class InferenceModule:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def summarize(self, model, tokenizer, text):
        prompt = f"Summarize the following text in less than four sentences:\n\n{text}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=self.cfg.max_target)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
