from datasets import load_dataset
import random
import json
import os

def download_data():
    # Let's use data from January 2024 as an example
    dataset = load_dataset("RealTimeData/bbc_news_alltime", "2024-01")
    return dataset["train"]

def build_context_continuation(dataset, output_path, num_samples=200):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    processed_samples = []
    for i, sample in enumerate(dataset):
        text = sample["content"] 
        if text and len(text) > 50:  # filter short text
            p = random.randint(20, len(text) - 20)  # pick a random point
            context = text[:p]
            continuation = text[p:]
            processed_samples.append({
                "context": context,
                "continuation": continuation
            })
        
        if len(processed_samples) >= num_samples:
            break
    
    # Save to disk
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main():
    dataset = download_data()
    build_context_continuation(dataset, "data/processed/bbc_context_pairs.jsonl")

if __name__ == "__main__":
    main()
