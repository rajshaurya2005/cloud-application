from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Any
import os

class VectorStoreInMemory:
    """A class for managing an in-memory FAISS vector store."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.db = None
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    def add_documents(self, chunks: List[Any]):
        """Add document chunks to the vector store."""
        if not chunks:
            print("No new document chunks to add.")
            return
        
        try:
            if self.db is None:
                self.db = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
                print(f"Created a new in-memory vector store and added {len(chunks)} chunks.")
            else:
                self.db.add_documents(documents=chunks)
                print(f"Added {len(chunks)} new chunks to the existing in-memory vector store.")

        except Exception as e:
            print(f"An error occurred while adding documents: {e}")
    
    def retriever(self, query: str, k: int = 5) -> List[Any]:
        """Retrieve similar documents from the vector store."""
        if self.db is None:
            raise ValueError("Vector store has not been initialized. Please add documents first.")
        
        try:
            similar_documents = self.db.similarity_search(query=query, k=k)
            return similar_documents
        except Exception as e:
            print(f"An error occurred during retrieval: {e}")
            return []

    def save_local(self, folder_path: str):
        """Save the FAISS index to a local folder."""
        if self.db is None:
            raise ValueError("Vector store is not initialized. Nothing to save.")
        
        try:
            os.makedirs(folder_path, exist_ok=True)
            self.db.save_local(folder_path)
            print(f"Vector store saved to {folder_path}")
        except Exception as e:
            print(f"An error occurred while saving the vector store: {e}")

    def load_local(self, folder_path: str):
        """Load a FAISS index from a local folder."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The specified folder path does not exist: {folder_path}")
            
        try:
            self.db = FAISS.load_local(folder_path, self.embeddings, allow_dangerous_deserialization=True)
            print(f"Vector store loaded from {folder_path}")
        except Exception as e:
            print(f"An error occurred while loading the vector store: {e}")
            raise e
