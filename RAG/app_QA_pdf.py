# Dependancies
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

import tempfile

# Page setup
st.set_page_config(
    page_title="PDF QA Assistant",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: "Times New Roman", serif;
    color: black;
}

.main {
    background-color: white;
}

h1, h2, h3, h4 {
    color: black;
    font-family: "Times New Roman", serif;
}

.stButton > button {
    background-color: #90EE90;
    color: black;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #7CFC00;
    color: black;
}

.stTextInput input {
    color: black;
}

section[data-testid="stSidebar"] {
    background-color: #f7f7f7;
}

</style>
""", unsafe_allow_html=True)

# Title
st.title("PDF Question Answering Assistant")

st.write(
    "Upload a PDF, provide your Groq API Key, and ask questions from the document."
)

# Sidebar
with st.sidebar:

    st.header("Configuration")

    groq_api_key = st.text_input(
        "Enter Groq API Key",
        type="password"
    )

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )
    
    # Process PDF
    def create_vectorstore(pdf_file):
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            pdf_path = tmp.name
            
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75
            )
        
        chunks = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectordb = FAISS.from_documents(
            chunks,
            embeddings)

        return vectordb
    
    # Build Vector Store
    
if uploaded_pdf and groq_api_key:
    
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    if "vectordb" not in st.session_state:
        with st.spinner("Processing PDF..."):
            st.session_state.vectordb = create_vectorstore(
                uploaded_pdf)

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0
            )

            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectordb.as_retriever()
            )

        st.success("PDF Processed Successfully!")

# Question Section
if "qa_chain" in st.session_state:

    st.subheader("Ask Questions")

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Get Answer"):

        if question:

            with st.spinner("Generating Answer..."):

                response = st.session_state.qa_chain.invoke(question)

                st.markdown("### Answer")

                st.write(response["result"])

else:
    st.info("Upload PDF and API Key to begin.")

