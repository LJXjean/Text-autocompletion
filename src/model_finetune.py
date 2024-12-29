# train.py

import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def load_data_for_finetuning(file_path):
    # Read JSON lines and prepare them for GPT2
    # Usually, you want to convert (context, continuation) into a single text sample with a special separator, or just "context + continuation"
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            text = item["context"] + "\n" + item["continuation"]
            data.append(text)
    return data

def main():
    model_name = "gpt2"  # or "gpt2-medium", "EleutherAI/gpt-neo-1.3B", etc.
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 1. Load data
    train_texts = load_data_for_finetuning("data/processed/bbc_context_pairs.jsonl")

    # 2. Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True)

    # Convert to dataset format for Trainer
    from datasets import Dataset
    raw_dataset = Dataset.from_dict({"text": train_texts})
    tokenized_dataset = raw_dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    
    # 3. Define data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Load model
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 5. Trainer config
    training_args = TrainingArguments(
        output_dir="./finetuned_model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=50,
        logging_steps=50,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # 7. Train
    trainer.train()
    trainer.save_model("./finetuned_model")

if __name__ == "__main__":
    main()
