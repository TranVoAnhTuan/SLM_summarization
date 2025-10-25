from data_loader import load_and_preprocess
from tokenizer_module import tokenize_datasets
from model_module import load_model
from trainer_module import train_model
from evaluate_module import evaluate_model
from inference_module import summarize_text

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    processed = load_and_preprocess()

    print("Tokenizing dataset...")
    tokenized, tokenizer = tokenize_datasets(processed)

    print("Loading model (local T5-small + LoRA)...")
    model = load_model()

    try:
        print("Starting training...")
        model = train_model(model, tokenized, tokenizer)
    except RuntimeError as e:
        print("GPU crashed, saving log...")
        with open("error_log.txt", "a") as f:
            f.write(str(e) + "\n")

    print("Evaluating model...")
    evaluate_model(model, tokenizer, processed)  

    print("Example inference:")
    text = """Nigeria's television survival show has been suspended after a contestant drowned in preparation for the program, said Dutch brewer Heineken's local unit which is sponsoring the show. Anthony Ogadje, 25, and nine other contestants had gone to Shere Hills Lake in Nigeria's hilly Plateau State to prepare for the "Gulder Ultimate Search," which sets a variety of physical challenges for participants. A statement from Nigerian Breweries on Monday said Ogadje died suddenly and he was thought to have drowned. "All attempts to revive him by the attendant medical team and the lifeguards, including his fellow contestants, failed," said Nigerian Breweries, which is majority-owned by the Dutch giant. Broadcasting had been due to start on Thursday. In the show, the weakest contestants are evicted one by one until a winner emerges. The prize money is a big attraction in a country where most people live in extreme poverty and benefit little from Nigeria's oil wealth. The winner was to get 5 million naira (about $39,000) in cash, a four-wheel drive jeep and another 500,000 naira (about $3,900) to buy clothes. The winner could also have expected to become an instant celebrity, attracting sponsorship deals. The Ultimate Search, which started in 2004, gets high ratings. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed."""
    summary = summarize_text(model, tokenizer, text)
    print(summary)
