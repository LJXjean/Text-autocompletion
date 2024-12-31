from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List

class AutocompleteModel:
    def __init__(self, model_path: str = "gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()

        # Set up padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def generate_text(
        self, 
        prompt: str, 
        retrieved_docs: List[str] = None, 
        max_length: int = 50, 
        temperature: float = 0.7, 
        top_p: float = 0.9
    ) -> str:
        # Combine prompt with retrieved docs if available
        if retrieved_docs:
            # Limit the length of retrieved docs
            combined_docs = ' '.join(retrieved_docs)[:500]
            context = f"{combined_docs}\n\nContext: {prompt}"
        else:
            context = prompt

        # Encode with strict length control
        inputs = self.tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # Limit input length
            padding=True
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,  # Add attention mask
                max_new_tokens=max_length,  # Use max_new_tokens instead of max_length
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1
            )
        
        # Decode only the new tokens
        original_length = inputs.input_ids.shape[1]
        generated_text = self.tokenizer.decode(
            outputs[0][original_length:],
            skip_special_tokens=True
        )
        return generated_text

