class Config:
    # Dataset
    dataset_name = "cnn_dailymail"
    dataset_version = "3.0.0"

    # Model
    model_name = "/home/jacktran/NLP/experiment/t5base_model/t5base_lora/chunk_20"
    output_dir = "/home/jacktran/NLP/experiment/t5base_model/t5base_lora"
    device = "cuda"

    # Tokenization
    max_input = 800
    max_target = 256
    num_proc = 4
    tokenized_cache = "./tokenized_cnn_dm"

    # Training
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 4
    learning_rate = 3e-4
    num_train_epochs = 2
    chunk_size = 10000

    # Evaluation
    num_eval_samples = None
    dataset_val = "test"
    eval_logs = "./evaluation_logs"
