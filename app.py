import streamlit as st
from pypdf import PdfReader
import pandas as pd
import base64
from datetime import datetime
import os
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text


# -----------------------------
# TEXT CHUNKING
# -----------------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# -----------------------------
# CREATE VECTOR STORE
# -----------------------------
def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# -----------------------------
# LOAD VECTOR STORE
# -----------------------------
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


# -----------------------------
# FORMAT RETRIEVED DOCS
# -----------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------
# CREATE QA CHAIN
# -----------------------------
def get_qa_chain(api_key, vectorstore):

    prompt_template = """
    Answer the question as detailed as possible using ONLY the provided context.
    If the answer is not present in the context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # ✅ Stable Gemini Model
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


# -----------------------------
# HANDLE QUESTION
# -----------------------------
def handle_user_question(user_question, api_key):
    if not os.path.exists("faiss_index"):
        return "Please upload and process PDFs first."

    try:
        vectorstore = load_vector_store()
        chain = get_qa_chain(api_key, vectorstore)

        response = chain.invoke(user_question)
        return response

    except Exception as e:
        error_msg = str(e)

        if "404" in error_msg:
            return "Model not found. Make sure Gemini API is enabled."
        elif "429" in error_msg:
            return "Rate limit exceeded. Try again later."
        elif "api" in error_msg.lower():
            return "Invalid API Key. Check your Google API key."
        else:
            return f"Error: {error_msg}"


# -----------------------------
# MAIN STREAMLIT APP
# -----------------------------
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.header("Chat with Multiple PDFs 📚")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar
    st.sidebar.title("Menu")

    api_key = st.sidebar.text_input(
        "Enter your Google API Key:",
        type="password"
    )

    st.sidebar.markdown(
        "Get API key from: https://makersuite.google.com/app/apikey"
    )

    pdf_docs = st.sidebar.file_uploader(
        "Upload PDF Files",
        accept_multiple_files=True,
        type=['pdf']
    )

    if st.sidebar.button("Submit & Process"):
        if not pdf_docs:
            st.sidebar.warning("Please upload PDFs.")
        elif not api_key:
            st.sidebar.warning("Please enter your Google API Key.")
        else:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text.strip():
                    st.sidebar.error("No text extracted from PDFs.")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    create_vector_store(text_chunks)
                    st.sidebar.success(
                        f"Processing complete! Created {len(text_chunks)} chunks."
                    )

    # Question Input
    user_question = st.text_input("Ask a Question from the PDFs")

    if user_question:
        if not api_key:
            st.warning("Please enter your API key.")
        else:
            with st.spinner("Generating answer..."):
                response = handle_user_question(user_question, api_key)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.session_state.conversation_history.append(
                    (user_question, response, timestamp)
                )

    # Display Chat History
    if st.session_state.conversation_history:
        st.markdown("### 💬 Chat")
        for question, answer, time_val in reversed(st.session_state.conversation_history):
            st.markdown(f"**You:** {question}")
            st.markdown(f"**Bot:** {answer}")
            st.markdown(f"_Time: {time_val}_")
            st.markdown("---")

    # Download Chat History
    if st.session_state.conversation_history:
        df = pd.DataFrame(
            st.session_state.conversation_history,
            columns=["Question", "Answer", "Timestamp"]
        )

        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="chat_history.csv">📥 Download Chat History</a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)


if __name__ == "__main__":
    main()