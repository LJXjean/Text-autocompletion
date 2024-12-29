import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGHelper:
    def __init__(self, docs, index_path="docs.index"):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.docs = docs
        self.index_path = index_path
        self.index = None

    def build_index(self):
        embeddings = self.embedder.encode(self.docs)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        faiss.write_index(self.index, self.index_path)

    def load_index(self):
        self.index = faiss.read_index(self.index_path)

    def search(self, query, top_k=3):
        q_vec = self.embedder.encode([query])
        distances, indices = self.index.search(q_vec, top_k)
        retrieved_texts = [self.docs[i] for i in indices[0]]
        return retrieved_texts
