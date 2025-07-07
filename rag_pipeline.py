from vector_store import VectorStoreInMemory
from document_loaders import DocumentParsers
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
import os

class Pipe:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model_name = model
        self.llm = ChatGroq(
            model=model,
            api_key=api_key
        )
        self.vector_store = VectorStoreInMemory()  # Create instance
        self.document_parser = DocumentParsers()
    
    def load_documents(self, file_paths, chunk_size=500, chunk_overlap=50):
        """Load and process documents into vector store"""
        all_chunks = []
        
        for file_path in file_paths:
            # Load document
            documents = self.document_parser.file_to_text(file_path)
            if documents:
                # Split into chunks
                chunks = self.document_parser.text_splitter(documents, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
        
        if all_chunks:
            # Add to vector store
            self.vector_store.add_documents(all_chunks)
            print(f"Loaded {len(all_chunks)} chunks into vector store")
        else:
            print("No documents were loaded")
    
    def load_existing_vector_store(self, index_path="faiss_index"):
        """Load existing vector store"""
        try:
            self.vector_store.load_local(index_path)
            print(f"Loaded existing vector store from {index_path}")
        except Exception as e:
            print(f"Could not load vector store: {e}")
            raise e
    
    def chain(self, query, k=5):
        """Main RAG chain"""
        try:
            # Get context from vector store
            context_docs = self.vector_store.retriever(query=query, k=k)
            
            if not context_docs:
                return "No relevant documents found for your query."
            
            # Format context
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Create messages
            system_prompt = SystemMessage(content="You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain enough information to answer the question, say so.")
            user_prompt = HumanMessage(content=f"""Based on the context below, answer the query to the best of your ability.

Context:
{context}

Query: {query}

Answer:""")
            
            # Get response from LLM
            messages = [system_prompt, user_prompt]
            result = self.llm.invoke(messages)
            
            return result.content
            
        except Exception as e:
            return f"Error processing query: {e}"

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipe = Pipe(
        api_key="your-groq-api-key", 
        model="llama-3.3-70b-versatile"
    )
    
    # Load documents (replace with your file paths)
    file_paths = ["document1.pdf", "document2.txt"]
    pipe.load_documents(file_paths, chunk_size=500, chunk_overlap=50)
    
    # Or load existing vector store
    # pipe.load_existing_vector_store("faiss_index")
    
    # Query the system
    response = pipe.chain("What is this document about?", k=3)
    print(response)
