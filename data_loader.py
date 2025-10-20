from datasets import load_dataset

def load_and_preprocess():
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    def preprocess(example):
        prompt = "Summarize the following text in less than four sentences:\n\n" + example["article"]
        return {"prompt": prompt, "summary": example["highlights"]}

    processed = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
    return processed
