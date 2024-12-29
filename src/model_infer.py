import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class AutocompleteModel:
    def __init__(self, model_path="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()
        # Ensure pad_token_id is set
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate_text(self, prompt, max_length=50, temperature=0.7, top_p=0.9):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=len(input_ids[0]) + max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )


        gen_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        return gen_text
