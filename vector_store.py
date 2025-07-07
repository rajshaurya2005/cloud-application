from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Any

class VectorStoreInMemory:
    def __init__(self) -> None:
        self.db = None
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def add_documents(self, chunks: List[Any]):
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
        if self.db is None:
            raise ValueError("Vector store has not been initialized. Please add documents first.")
        
        try:
            similar_documents = self.db.similarity_search(query=query, k=k)
            return similar_documents
        except Exception as e:
            print(f"An error occurred during retrieval: {e}")
            return []
