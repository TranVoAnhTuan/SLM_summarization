def summarize_text(model, tokenizer, text):
    prompt = f"Summarize the following text in less than four sentences:\n\n{text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
