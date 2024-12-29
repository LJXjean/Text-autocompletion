from fastapi import FastAPI
from pydantic import BaseModel
from model_infer import AutocompleteModel
# from rag_utils import RAGHelper  # if you use RAG

app = FastAPI()
autocomplete_model = AutocompleteModel("gpt2")  # or "gpt2" if not fine-tuned

class AutocompleteRequest(BaseModel):
    text_before_cursor: str
    max_length: int = 50

@app.post("/autocomplete")
def autocomplete(req: AutocompleteRequest):
    try:
        prompt = req.text_before_cursor

        # (Optional) RAG
        # retrieved_docs = rag_helper.search(req.text_before_cursor, top_k=3)
        # prompt = f"{prompt}\nRelevant info: {' '.join(retrieved_docs)}\n"

        completion = autocomplete_model.generate_text(
            prompt,
            max_length=req.max_length,
            temperature=0.7,
            top_p=0.9
        )

        return {"completion": completion}
    except Exception as e:
        return {"error": str(e)}
