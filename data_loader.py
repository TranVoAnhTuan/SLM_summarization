from datasets import load_dataset
from config import Config

class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load_and_preprocess(self):
        print(f"Loading dataset {self.cfg.dataset_name} ...")
        dataset = load_dataset(self.cfg.dataset_name, self.cfg.dataset_version)

        def preprocess(example):
            prompt = "Summarize the following text in less than four sentences:\n\n" + example["article"]
            return {"prompt": prompt, "summary": example["highlights"]}

        processed = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
        print("Dataset loaded and preprocessed.")
        return processed
