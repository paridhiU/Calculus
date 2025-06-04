import streamlit as st
from rag_chain import get_rag_chain
from wolfram import query_wolfram
from gemini_llm import ask_gemini
import re
import zipfile
import os

def unzip_faiss_index():
    zip_path = "faiss_index.zip"
    extract_dir = "faiss_index"

    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

st.set_page_config(page_title="Calculus Q&A", layout="wide")
st.title("📚 Ask Calculus")

# Load the RAG chain
qa_chain = get_rag_chain()

# Function to check if a query is math-related
def is_math_query(query):
    math_keywords = [
        "integral", "derivative", "solve", "limit", "∫", "∑", "dx", "=",
        "^", "roots", "simplify", "differentiate", "calculate", "cos", "sin", "log"
    ]
    return any(k in query.lower() for k in math_keywords) or bool(re.search(r"[0-9x=^*/+\-]", query))

# Core logic for routing the question

def answer_query(query):
    if is_math_query(query):
        # Try Wolfram Alpha first
        try:
            wolfram_answer = query_wolfram(query)
            if wolfram_answer and "No result" not in wolfram_answer:
                return {"answer": wolfram_answer}
        except Exception as e:
            print(f"Wolfram error: {e}")

        # Fallback to Gemini
        try:
            gemini_answer = ask_gemini(query)
            if gemini_answer and gemini_answer.strip():
                return {"answer": gemini_answer}
            else:
                print("Gemini response is empty or None.")
        except Exception as e:
            print(f"Gemini error: {e}")

        return {"answer": "Sorry, both Wolfram Alpha and Gemini could not process your question."}

    # Fallback to RAG for theory questions
    try:
        rag_result = qa_chain(query)
        return {
            "answer": rag_result["result"],
            "sources": rag_result.get("source_documents", [])
        }
    except Exception as e:
        return {"answer": f"An error occurred while processing the query with Groq+RAG: {e}"}


# Input box
query = st.text_input("💬 Ask a Calculus Question:")

if query:
    with st.spinner("Thinking..."):
        result = answer_query(query)

        # Display the answer
        st.markdown("### 🧠 Answer:")
        st.write(result["answer"])

        # Show source documents only if present
        if result.get("sources"):
            with st.expander("📄 Sources"):
                for doc in result["sources"]:
                    source = doc.metadata.get("source", "N/A")
                    preview = doc.page_content[:500] + "..."
                    st.markdown(f"**Source:** {source}")
                    st.write(preview)
