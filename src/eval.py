# eval.py
import json
import os
import random
from typing import List

import nltk
from nltk.translate.bleu_score import sentence_bleu

from rouge_score import rouge_scorer

# Local imports: your model and possibly RAG helper
from model_infer import AutocompleteModel
from rag_utils import RAGHelper

# ======================
# CONFIG
# ======================

TEST_FILE = "../data/processed/bbc_context_pairs_eval.jsonl"
MODEL_PATH = "gpt2"      # or fine-tuned model, e.g., "./finetuned_model"
USE_RAG = True           # set True if you want to retrieve docs before generating
TOP_K_DOCS = 3           # how many docs to retrieve
NUM_SAMPLES = 50         # how many samples to evaluate (subset of your test set)
MAX_GEN_LEN = 50         # how many tokens/words to generate

# ======================
# METRIC UTILS
# ======================

def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    A simple function to compute sentence-level BLEU using NLTK.
    """
    # Tokenize
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    # Compute BLEU
    # Here we do a single-reference BLEU. For multi-reference, pass list of lists.
    bleu_score = sentence_bleu([ref_tokens], hyp_tokens)
    return bleu_score

def compute_rouge(reference: str, hypothesis: str) -> dict:
    """
    Compute ROUGE-L (and possibly others) using the rouge_score library.
    Returns a dict of scores.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure
    }

# ======================
# MAIN EVAL ROUTINE
# ======================

def load_test_data(file_path: str, num_samples: int = 50) -> List[dict]:
    """
    Load up to `num_samples` lines from the JSONL test file,
    each containing {"context": ..., "continuation": ...}.
    """
    samples = []
    if not os.path.exists(file_path):
        print(f"Test file {file_path} not found!")
        return samples

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "context" in obj and "continuation" in obj:
                samples.append(obj)
    
    # Shuffle and take subset
    random.shuffle(samples)
    return samples[:num_samples]

def main():
    # 1. Load the test data
    test_samples = load_test_data(TEST_FILE, NUM_SAMPLES)
    if not test_samples:
        print("No test samples found. Exiting...")
        return
    
    print(f"Loaded {len(test_samples)} test samples from {TEST_FILE}.")

    # 2. Initialize model
    print(f"Loading model: {MODEL_PATH}")
    model = AutocompleteModel(MODEL_PATH)

    # 3. (Optional) Initialize RAG
    if USE_RAG:
        rag_helper = RAGHelper(docs_dir="data/docs")
    else:
        rag_helper = None

    # 4. Metrics accumulators
    total_bleu = 0.0
    total_rouge1 = 0.0
    total_rougeL = 0.0

    # Keep track of (context, ref, pred) for a few examples
    examples_to_show = []

    # 5. Inference + metric calculation
    for i, sample in enumerate(test_samples):
        context = sample["context"]
        reference = sample["continuation"]

        # (Optional) retrieve docs
        retrieved_docs_str = []
        if rag_helper:
            retrieved_docs = rag_helper.search(context, top_k=TOP_K_DOCS)
            # doc.page_content is the actual text
            retrieved_docs_str = [doc.page_content for doc in retrieved_docs]

        # Generate text from the model
        predicted = model.generate_text(
            prompt=context,
            retrieved_docs=retrieved_docs_str,
            max_length=MAX_GEN_LEN,
            temperature=0.7,
            top_p=0.9
        )

        # Compute metrics
        bleu_score = compute_bleu(reference, predicted)
        rouge_scores = compute_rouge(reference, predicted)

        total_bleu += bleu_score
        total_rouge1 += rouge_scores["rouge1"]
        total_rougeL += rouge_scores["rougeL"]

        # Optionally store some random examples to print later
        if i < 3:  # just store first 3 examples, or pick randomly
            examples_to_show.append({
                "context": context[:100] + "...",  # truncated for display
                "reference": reference[:100] + "...",
                "prediction": predicted[:100] + "..."
            })

    # 6. Compute average metric
    n = len(test_samples)
    avg_bleu = total_bleu / n
    avg_rouge1 = total_rouge1 / n
    avg_rougeL = total_rougeL / n

    # 7. Print results
    print("============ EVALUATION RESULTS ============")
    print(f"Number of samples evaluated: {n}")
    print(f"Average BLEU:    {avg_bleu:.4f}")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-L: {avg_rougeL:.4f}")
    print("============================================\n")

    print("Sample predictions:")
    for idx, ex in enumerate(examples_to_show):
        print(f"\n--- Example {idx+1} ---")
        print(f"Context:    {ex['context']}")
        print(f"Reference:  {ex['reference']}")
        print(f"Prediction: {ex['prediction']}")
    print("============================================")

if __name__ == "__main__":
    main()
