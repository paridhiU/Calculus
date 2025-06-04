import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_rag_chain():
    # Load embedding model
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Load FAISS vector store index from local disk
    try:
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index: {e}")

    # Create a retriever from vector store, customize number of docs to retrieve
    retriever = db.as_retriever(search_kwargs={"k": 5})  # adjust k as needed

    # Initialize Groq Chat LLM
    model_name = os.getenv("GROQ_MODEL", "mistral-saba-24b")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    llm = ChatGroq(
        model_name=model_name,
        api_key=api_key,
        temperature=0.2  # lower temperature for more precise answers (adjust as needed)
    )

    # Create RetrievalQA chain with retriever and LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
