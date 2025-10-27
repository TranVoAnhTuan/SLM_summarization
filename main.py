from config import Config
from data_loader import DataLoader
from tokenizer_module import TokenizerModule
from model_module import ModelLoader
from trainer_module import TrainerModule
from evaluate_module import Evaluator
from inference_module import InferenceModule

def main():
    cfg = Config()

    data_loader = DataLoader(cfg)
    processed = data_loader.load_and_preprocess()

    tokenizer_module = TokenizerModule(cfg)
    tokenized, tokenizer = tokenizer_module.tokenize(processed)

    model_loader = ModelLoader(cfg)
    model = model_loader.load_model()

    trainer = TrainerModule(cfg)
    model = trainer.train(model, tokenized, tokenizer)

    evaluator = Evaluator(cfg)
    evaluator.evaluate(model, tokenizer, processed)

    infer = InferenceModule(cfg)
    test_text = "Summarize the following text in less than four sentences: Nigeria's television survival show has been suspended ..."
    summary = infer.summarize(model, tokenizer, test_text)
    print("\nGenerated Summary:\n", summary)

if __name__ == "__main__":
    main()
