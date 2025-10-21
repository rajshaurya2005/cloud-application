from vector_store import VectorStoreInMemory
from document_loaders import DocumentParsers
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from typing import List

class Pipe:
    """A Retrieval-Augmented Generation (RAG) pipeline."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model_name = model
        self.llm = ChatGroq(model=model, api_key=api_key)
        self.vector_store = VectorStoreInMemory()
        self.document_parser = DocumentParsers()
    
    def load_documents(self, file_paths: List[str], chunk_size: int = 500, chunk_overlap: int = 50):
        """Load and process documents into the vector store."""
        all_chunks = []
        
        for file_path in file_paths:
            documents = self.document_parser.file_to_text(file_path)
            if documents:
                chunks = self.document_parser.text_splitter(documents, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
        
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            print(f"Loaded {len(all_chunks)} chunks into vector store.")
        else:
            print("No documents were loaded.")
    
    def save_vector_store(self, index_path: str = "faiss_index"):
        """Save the vector store to a local path."""
        self.vector_store.save_local(index_path)

    def load_existing_vector_store(self, index_path: str = "faiss_index"):
        """Load an existing vector store from a local path."""
        self.vector_store.load_local(index_path)
    
    def chain(self, query: str, k: int = 5) -> str:
        """The main RAG chain for processing queries."""
        try:
            context_docs = self.vector_store.retriever(query=query, k=k)
            
            if not context_docs:
                return "No relevant documents found for your query."
            
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            system_prompt = SystemMessage(content="You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain enough information, say so.")
            user_prompt = HumanMessage(content=f"""Based on the context below, answer the query to the best of your ability.

Context:
{context}

Query: {query}

Answer:""")
            
            messages = [system_prompt, user_prompt]
            result = self.llm.invoke(messages)
            
            return result.content
            
        except Exception as e:
            return f"Error processing query: {e}"

# Example usage
if __name__ == "__main__":
    # This is an example, do not hardcode your API key in production
    pipe = Pipe(
        api_key="YOUR_GROQ_API_KEY", 
        model="llama3-8b-8192"
    )
    
    # Example file paths
    file_paths = [] # Add paths to your documents, e.g., ["./document1.pdf"]
    if file_paths:
        pipe.load_documents(file_paths, chunk_size=500, chunk_overlap=50)
        
        # Save the vector store
        # pipe.save_vector_store("my_faiss_index")
    else:
        print("No file paths provided for loading.")

    # Or load an existing vector store
    # try:
    #     pipe.load_existing_vector_store("my_faiss_index")
    # except Exception as e:
    #     print(f"Could not load existing vector store: {e}")

    # Query the system if the vector store is loaded
    if pipe.vector_store.db:
        response = pipe.chain("What is this document about?", k=3)
        print("\n--- Response ---")
        print(response)
        print("----------------\n")
    else:
        print("Vector store is not initialized. Please load documents first.")
