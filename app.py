import streamlit as st
import tempfile
import os
from pathlib import Path
import shutil

# Import your fixed classes
from rag_pipeline import Pipe

def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory"""
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        # Save file to temp directory
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    return file_paths, temp_dir

def initialize_rag_pipeline(api_key, model):
    """Initialize RAG pipeline and store in session state"""
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = Pipe(api_key=api_key, model=model)
    return st.session_state.rag_pipeline

def rag_answer(query: str, api_key: str, model: str, files, chunk_size: int, chunk_overlap: int) -> str:
    """Process RAG query with actual implementation"""
    try:
        # Initialize pipeline
        pipe = initialize_rag_pipeline(api_key, model)
        
        # Process uploaded files if new files are uploaded
        if files and "processed_files" not in st.session_state:
            with st.spinner("Processing uploaded documents..."):
                file_paths, temp_dir = save_uploaded_files(files)
                pipe.load_documents(file_paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                # Clean up temp directory
                shutil.rmtree(temp_dir)
                
                # Mark files as processed
                st.session_state.processed_files = [f.name for f in files]
                st.success("Documents processed successfully!")
        
        # Check if we have documents loaded
        if pipe.vector_store.db is None:
            try:
                pipe.load_existing_vector_store()
            except:
                return "⚠️ No documents found. Please upload some files first and wait for them to be processed."
        
        # Get answer
        response = pipe.chain(query, k=5)
        return response
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Streamlit app configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
    }
    .file-info {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>💬 RAG Chatbot</h1><p>Ask questions about your uploaded documents!</p></div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔧 Settings")

# API Key input
api_key = st.sidebar.text_input("Groq API Key", type="password", key="api_key", 
                               help="Get your API key from https://console.groq.com/keys")

# Model selection
model = st.sidebar.selectbox("Model", 
                            ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"], 
                            key="model")

# File upload
uploaded_files = st.sidebar.file_uploader(
    "Upload files", 
    accept_multiple_files=True,
    type=['pdf', 'txt', 'csv'],
    help="Upload PDF, TXT, or CSV files"
)

# Chunk settings
st.sidebar.subheader("📄 Chunk Settings")
chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=2000, value=500, step=100,
                              help="Size of text chunks for processing")
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=50, step=10,
                                 help="Overlap between consecutive chunks")

# Reset button
if st.sidebar.button("🔄 Reset Session"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Display file info
if uploaded_files:
    st.sidebar.success(f"📁 {len(uploaded_files)} file(s) uploaded")
    with st.sidebar.expander("View uploaded files"):
        for file in uploaded_files:
            file_size = len(file.getbuffer()) / 1024  # Size in KB
            st.write(f"📄 {file.name} ({file_size:.1f} KB)")
    
    # Check if files have changed
    current_files = [f.name for f in uploaded_files]
    if "processed_files" in st.session_state and st.session_state.processed_files != current_files:
        # Files have changed, reset processed status
        if "processed_files" in st.session_state:
            del st.session_state.processed_files
        st.sidebar.info("New files detected. They will be processed with your next query.")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 💭 Chat with your documents")

with col2:
    if uploaded_files and "processed_files" in st.session_state:
        st.success("✅ Ready")
    elif uploaded_files:
        st.warning("⏳ Processing...")
    else:
        st.error("📄 No files")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about your documents...", key="chat_input"):
    # Validation
    if not api_key:
        st.error("⚠️ Please provide your Groq API key in the sidebar.")
        st.stop()
    
    if not uploaded_files:
        st.error("⚠️ Please upload at least one document first.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_answer(
                prompt,
                api_key,
                model,
                uploaded_files,
                chunk_size,
                chunk_overlap
            )
            st.markdown(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

# Instructions
with st.expander("ℹ️ How to use this RAG Chatbot"):
    st.markdown("""
    ### Getting Started:
    1. **🔑 Add your Groq API key** in the sidebar
       - Get it from [console.groq.com/keys](https://console.groq.com/keys)
    2. **📄 Upload your documents** (PDF, TXT, CSV supported)
    3. **⚙️ Adjust chunk settings** if needed (optional)
    4. **💬 Ask questions** about your documents
    
    ### Features:
    - **Multiple file support**: Upload multiple documents at once
    - **Persistent chat**: Your conversation history is maintained
    - **Configurable chunking**: Adjust how documents are split
    - **Multiple models**: Choose from different LLM models
    
    ### Tips:
    - Start with specific questions about your documents
    - If answers seem incomplete, try rephrasing your question
    - Use the reset button to clear session and start fresh
    """)

# Footer
st.markdown("---")
st.markdown("**Built with** 🔗 Streamlit • 🦜 LangChain • ⚡ Groq")

# Display current status
if st.session_state.get("processed_files"):
    st.sidebar.markdown("### 📊 Status")
    st.sidebar.success(f"✅ {len(st.session_state.processed_files)} files processed")
    st.sidebar.info(f"🔧 Model: {model}")
    st.sidebar.info(f"📏 Chunk size: {chunk_size}")
