# rag_utils.py

import json
import glob
import os
from typing import List, Optional

# LangChain imports for RAG
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class RAGHelper:
    """
    A helper class to load .jsonl documents from a directory, build a FAISS vector store,
    and provide a 'search' method for retrieval-augmented generation.
    """
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  
        self.vectorstore = self._build_vectorstore(docs_dir)

    def _build_vectorstore(self, docs_dir: str) -> Optional[FAISS]:
        """Collects text from .jsonl files, splits into chunks, then builds a FAISS store."""
        doc_texts = []

        # Find all .jsonl files recursively
        jsonl_files = glob.glob(os.path.join(docs_dir, "**/*.jsonl"), recursive=True)
        if not jsonl_files:
            print(f"No JSONL files found in {docs_dir}. Vector store will be empty.")
            return None

        # Read each file line by line, extract 'content' or 'text' field
        for file_path in jsonl_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        content = data.get("content") or data.get("text")
                        if content and len(content) > 0:
                            doc_texts.append(content)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

        if not doc_texts:
            print("No valid text found. Vector store will be empty.")
            return None

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        docs = text_splitter.create_documents(doc_texts)

        # Build FAISS store
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        return vectorstore

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """Return the top-k most similar Document objects."""
        if not self.vectorstore:
            return []
        return self.vectorstore.similarity_search(query, k=top_k)
