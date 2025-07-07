from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentParsers:
    def file_to_text(self, file_path):    
        mapping = {
            ".pdf": PyPDFLoader(file_path),
            ".txt": TextLoader(file_path),
            ".csv": CSVLoader(file_path),
        }
        
        for ext, loader in mapping.items():
            if file_path.endswith(ext):
                try:
                    documents = loader.load()
                    return documents
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    return None
        
        print(f"Unsupported file type: {file_path}")
        return None

    def text_splitter(self, documents, chunk_size, chunk_overlap):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        return chunks
