from config import Config
from data_loader import DataLoader
from tokenizer_module import TokenizerModule
from model_module import ModelLoader
from trainer_module import TrainerModule
from evaluate_module import Evaluator
from inference_module import InferenceModule
from translate_module import TranslateModule

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

    translate = TranslateModule()
    text_vi = "Bộ Y tế vừa công bố kế hoạch phòng chống dịch bệnh mùa đông."
    text_en = translate.vi_to_en(text_vi)
    summary = infer.summarize(model, tokenizer, text_en)
    text_en = translate.en_to_vi(summary)
    print("\nTóm tắt tiếng Việt:\n",text_en)

if __name__ == "__main__":
    main()
