# Run the server with:
#   python -m uvicorn server:app --host 0.0.0.0 --port 8000

from fastapi import FastAPI
from pydantic import BaseModel

# Local imports
from model_infer import AutocompleteModel  # You must implement or import this
from rag_utils import RAGHelper           # See rag_utils.py below

# Initialize FastAPI app
app = FastAPI()

# Global model instance (GPT-2 or a fine-tuned variant)
autocomplete_model = AutocompleteModel("gpt2")

# Global RAG helper to build & query the FAISS vector store
rag_helper = RAGHelper(docs_dir="data/processed")

class AutocompleteRequest(BaseModel):
    text_before_cursor: str
    max_length: int = 50

@app.post("/autocomplete")
def autocomplete(req: AutocompleteRequest):
    """Endpoint to perform text autocompletion with optional RAG retrieval."""
    try:
        # 1) Retrieve top-k relevant docs from FAISS
        retrieved_docs = rag_helper.search(req.text_before_cursor, top_k=3)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]  # doc.page_content is the actual chunk text

        # 2) Call the language model's generate method
        generated_text = autocomplete_model.generate_text(
            prompt=req.text_before_cursor,
            retrieved_docs=retrieved_texts,
            max_length=req.max_length,
            temperature=0.7,
            top_p=0.9
        )

        return {"completion": generated_text}
    except Exception as e:
        return {"error": str(e)}
