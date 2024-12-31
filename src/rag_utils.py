# rag_utils.py

import json
import glob
import os
from typing import List, Optional

# LangChain imports for RAG
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class RAGHelper:
    """
    A helper class to load .jsonl documents from a directory, build a FAISS vector store,
    and provide a 'search' method for retrieval-augmented generation.
    """
    def __init__(self, docs_dir: str):
        self.docs_dir = docs_dir
        self.embeddings = HuggingFaceEmbeddings()
        
        # Split text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced from 1000
            chunk_overlap=50  # Reduced from 200
        )
        
        # Initialize vector store and assign it to self.vectorstore
        self.vectorstore = self.initialize_vectorstore(docs_dir, text_splitter)

    def initialize_vectorstore(self, docs_dir, text_splitter):
        """Collects text from .jsonl files, splits into chunks, then builds a FAISS store."""
        doc_texts = []

        # Find all .jsonl files recursively
        jsonl_files = glob.glob(os.path.join(docs_dir, "**/*.jsonl"), recursive=True)
        if not jsonl_files:
            print(f"No JSONL files found in {docs_dir}. Vector store will be empty.")
            # Return empty FAISS index instead of None
            return FAISS.from_texts([""], self.embeddings)

        # Read each file line by line
        for file_path in jsonl_files:
            print(f"Processing file: {file_path}")  # Debug print
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Check for both 'context' and 'content' fields
                        content = data.get("context") or data.get("content") or data.get("text")
                        if content and isinstance(content, str) and len(content.strip()) > 0:
                            doc_texts.append(content.strip())
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {file_path}: {e}")
                        continue

        if not doc_texts:
            print("No valid text found in files. Creating empty vector store.")
            return FAISS.from_texts([""], self.embeddings)

        print(f"Found {len(doc_texts)} documents to process")  # Debug print
        
        # Split text into chunks
        docs = text_splitter.create_documents(doc_texts)
        print(f"Created {len(docs)} chunks after splitting")  # Debug print

        # Build FAISS store
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        return vectorstore

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """Return the top-k most similar Document objects."""
        if not hasattr(self, 'vectorstore') or self.vectorstore is None:
            print("Warning: Vector store is not initialized")
            return []
        return self.vectorstore.similarity_search(query, k=top_k)
