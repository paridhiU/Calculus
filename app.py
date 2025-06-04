import streamlit as st
from rag_chain import get_rag_chain
from wolfram import query_wolfram
from gemini_llm import ask_gemini
import re


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
def looks_like_idk(answer):
    idk_phrases = [
        "i don't know", "i am not sure", "unable to find", "couldn't find", 
        "no relevant", "not confident", "i'm not certain", "sorry", "unknown"
    ]
    return any(phrase in answer.lower() for phrase in idk_phrases) or len(answer.strip()) < 30
def answer_query(query):
    # If math, try Wolfram then Gemini
    if is_math_query(query):
        try:
            wolfram_answer = query_wolfram(query)
            if wolfram_answer and "No result" not in wolfram_answer:
                return {"answer": wolfram_answer}
        except:
            pass
        try:
            gemini_answer = ask_gemini(query)
            return {"answer": gemini_answer}
        except:
            pass

    # Else use RAG (Groq + PDF)
    rag_result = qa_chain(query)
    answer = rag_result["result"]

    # Fallback to Gemini if Groq RAG fails
    if looks_like_idk(answer):
        try:
            gemini_answer = ask_gemini(query)
            return {"answer": gemini_answer}
        except:
            pass

    return {
        "answer": answer,
        "sources": rag_result.get("source_documents", [])
    }


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
