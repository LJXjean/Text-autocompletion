from datasets import load_dataset, concatenate_datasets
import random
import json
import os

def download_data():
    """
    Using BBC news data from January, June and December 2024.
    Loads and combines data from all three months.
    """
    # Load datasets for all months
    dataset_jan = load_dataset("RealTimeData/bbc_news_alltime", "2024-01")
    dataset_jun = load_dataset("RealTimeData/bbc_news_alltime", "2024-06")
    dataset_dec = load_dataset("RealTimeData/bbc_news_alltime", "2024-12")
    
    # Select only the content column from each dataset
    dataset_jan = dataset_jan["train"].select_columns(['content'])
    dataset_jun = dataset_jun["train"].select_columns(['content'])
    dataset_dec = dataset_dec["train"].select_columns(['content'])
    
    # Combine the datasets
    combined_dataset = concatenate_datasets([dataset_jan, dataset_jun, dataset_dec])
    return combined_dataset

def build_context_continuation_splits(
    dataset, 
    train_output_path, 
    eval_output_path, 
    train_samples=200, 
    eval_samples=50
):
    """
    Creates two files:
      - train_output_path: up to train_samples (context, continuation) pairs
      - eval_output_path: up to eval_samples (context, continuation) pairs
    """

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)

    # Convert HuggingFace Dataset to a list so we can shuffle or iterate easily
    data_list = list(dataset)

    # Shuffle data to avoid picking only the earliest or latest articles
    random.shuffle(data_list)

    train_records = []
    eval_records = []

    # Build (context, continuation) pairs
    for sample in data_list:
        text = sample.get("content", "")  # or "text" depending on dataset columns
        if text and len(text) > 50:
            # Pick a random point in the text
            p = random.randint(20, len(text) - 20)
            context = text[:p]
            continuation = text[p:]

            # If train_records still needs more samples, go there first
            if len(train_records) < train_samples:
                train_records.append({
                    "context": context,
                    "continuation": continuation
                })
            # Else if we still need eval samples, go there next
            elif len(eval_records) < eval_samples:
                eval_records.append({
                    "context": context,
                    "continuation": continuation
                })
            
            # If we've reached both limits, we can break early
            if len(train_records) >= train_samples and len(eval_records) >= eval_samples:
                break

    # Save train samples
    with open(train_output_path, 'w', encoding='utf-8') as f:
        for item in train_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save eval samples
    with open(eval_output_path, 'w', encoding='utf-8') as f:
        for item in eval_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    # 1. Download or load your BBC dataset
    dataset = download_data()

    # 2. Build the train & eval splits
    build_context_continuation_splits(
        dataset,
        train_output_path="data/processed/bbc_context_pairs_train.jsonl",
        eval_output_path="data/processed/bbc_context_pairs_eval.jsonl",
        train_samples=5000,
        eval_samples=1000
    )

if __name__ == "__main__":
    main()
