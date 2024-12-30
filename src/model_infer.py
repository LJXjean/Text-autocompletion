from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List

class AutocompleteModel:
    def __init__(self, model_path: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()

    def generate_text(
        self, 
        prompt: str, 
        retrieved_docs: List[str], 
        max_length: int = 50, 
        temperature: float = 0.7, 
        top_p: float = 0.9
    ) -> str:
        # Optionally incorporate retrieved_docs into your prompt
        combined_prompt = prompt + "\n\nRelevant:\n" + "\n".join(retrieved_docs) + "\n\nContinue:\n"

        input_ids = self.tokenizer.encode(combined_prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
