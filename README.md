# 💬 RAG Chatbot with Streamlit and Groq

This project is a versatile Retrieval-Augmented Generation (RAG) application built with Python. It uses the Streamlit framework for the user interface, LangChain for the RAG pipeline, Groq for fast LLM inference, and FAISS for in-memory vector storage.

## ✨ Features

- **Intuitive UI**: A simple and clean user interface built with Streamlit.
- **Multiple File Types**: Upload and process `.pdf`, `.txt`, and `.csv` files.
- **Fast Inference**: Powered by the Groq LPU™ Inference Engine for real-time responses.
- **Configurable**: Easily adjust chunking and retrieval settings from the UI.
- **In-Memory Vector Store**: Uses FAISS for efficient in-memory similarity searches.
- **Session Management**: Chat history is preserved during a session, and you can easily reset it.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **RAG & LLM Integration**: LangChain, LangChain Community, LangChain-Groq
- **LLM Provider**: Groq
- **Vector Store**: FAISS (via `faiss-cpu`)
- **Embeddings**: Hugging Face Sentence Transformers
- **File Parsing**: PyPDF2

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A Groq API key. You can get one from the [Groq Console](https://console.groq.com/keys).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/rajshaurya2005/cloud-application.git](https://github.com/rajshaurya2005/cloud-application.git)
    cd cloud-application
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## usage How to Use

1.  **Enter your Groq API Key** in the settings panel on the left.
2.  **Upload one or more documents** (`.pdf`, `.txt`, or `.csv`).
3.  Once the files are uploaded, **ask a question** in the chat input at the bottom of the page.
4.  The application will process the documents and use the information to answer your question.

## 📝 To-Do / Future Enhancements

- [ ] Add support for more file formats (e.g., `.docx`, `.pptx`).
- [ ] Implement persistent storage for the vector store across sessions.
- [ ] Add user authentication.
- [ ] Containerize the application with Docker.
