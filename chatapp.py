import streamlit as st
import os
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

from transformers import pipeline


# -------------------- STREAMLIT SESSION STATE --------------------
if "processed" not in st.session_state:
    st.session_state.processed = False


# -------------------- PDF TEXT EXTRACTION --------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# -------------------- TEXT CHUNKING --------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# -------------------- EMBEDDINGS (CACHED) --------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -------------------- VECTOR STORE --------------------
def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("❌ No readable text found in the uploaded PDFs.")
        return

    embeddings = load_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# -------------------- LLM (CACHED & FIXED) --------------------
@st.cache_resource
def load_llm():
    hf_pipeline = pipeline(
        "text2text-generation",        # correct for FLAN-T5
        model="google/flan-t5-base",
        max_new_tokens=256,             # output length only
        truncation=True                 # prevent token overflow
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


# -------------------- QA CHAIN --------------------
def get_conversational_chain():
    prompt_template = """
    Answer the question using ONLY the provided context.
    If the answer is not available, say:
    "Answer is not available in the provided documents."

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

    return load_qa_chain(
        llm=load_llm(),
        chain_type="stuff",
        prompt=prompt
    )


# -------------------- USER QUESTION --------------------
def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.warning("⚠️ Please upload PDFs and click 'Submit & Process' first.")
        return

    embeddings = load_embeddings()
    db = FAISS.load_local("faiss_index", embeddings)

    # ✅ limit retrieved chunks to avoid long context
    docs = db.similarity_search(user_question, k=3)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("### 🤖 Reply")
    st.write(response["output_text"])


# -------------------- MAIN APP --------------------
def main():
    st.set_page_config(
        page_title="Multi-PDF ChatApp – Priyanka Pareek",
        page_icon="📚"
    )

    st.header("📚 Multi-PDF Chat Agent 🤖")
    st.caption("Developed by **Priyanka Pareek**")

    user_question = st.text_input("Ask a question from the uploaded PDF documents ✍️")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.image("img/Robot.jpg")
        st.write("---")

        st.title("📁 PDF Upload Section")
        pdf_docs = st.file_uploader(
            "Upload PDF files and click **Submit & Process**",
            accept_multiple_files=True
        )

        # reset processing if new PDFs uploaded
        if pdf_docs:
            st.session_state.processed = False

        if st.button("Submit & Process") and not st.session_state.processed:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.processed = True
                st.success("PDFs processed successfully ✅")

        st.write("---")
        st.write("👩‍💻 **AI App created by Priyanka Pareek**")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%;
        background-color: #0E1117; padding: 12px; text-align: center;">
        © 2025 <a href="https://github.com/priyanka10111" target="_blank">
        Priyanka Pareek</a> | Built with ❤️ using Streamlit & LangChain
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
