# app.py (Streamlit app)
import streamlit as st
from src.ocr.ocr_processor import extract_text_from_image
from src.indexing.indexer import index_document
from src.retrieval.retriever import search_case_laws
from src.qa.qa_engine import setup_qa_engine, answer_question
from elasticsearch import Elasticsearch
import os
import io

# Setup Elasticsearch
es_host = os.environ.get("ELASTIC_HOST", "localhost")
es_port = int(os.environ.get("ELASTIC_PORT", 9200))
es = Elasticsearch([{'host': es_host, 'port': es_port}])

# Setup QA Engine (initialize once)
@st.cache_resource
def load_qa_engine():
    model_path = "models/llama-2-7b-chat.Q4_K_M.gguf"  # Replace with your model path
    return setup_qa_engine(es, "legal_docs", model_path)

qa = load_qa_engine()

st.title("Legal Document Analyzer")

# File Upload
uploaded_file = st.file_uploader("Upload a legal document (image)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # OCR Processing
    st.subheader("Extracted Text:")
    try:
        image_bytes = uploaded_file.getvalue()
        image_stream = io.BytesIO(image_bytes)
        text = extract_text_from_image(image_stream)

        if text:
            st.write(text)

            # Indexing
            if st.button("Index Document"):
                if index_document(es, "legal_docs", uploaded_file.name, text):
                    st.success("Document indexed successfully.")
                else:
                    st.error("Failed to index document.")

            # Case Law Retrieval
            query = st.text_input("Enter a query to search case laws:")
            if st.button("Search Case Laws"):
                results = search_case_laws(es, "legal_cases", query)
                if results:
                    st.subheader("Case Law Results:")
                    for result in results:
                        st.write(result)
                else:
                    st.write("No matching case laws found.")

            # Question Answering
            question = st.text_input("Ask a legal question:")
            if st.button("Answer Question"):
                answer = answer_question(qa, question)
                if answer:
                    st.subheader("Answer:")
                    st.write(answer)
                else:
                    st.write("Could not answer the question.")
        else:
            st.error("Error during OCR processing.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# src/ocr/ocr_processor.py
import pytesseract
from PIL import Image
import io

def extract_text_from_image(image_stream):
    """Extracts text from an image stream using Tesseract OCR."""
    try:
        img = Image.open(image_stream)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return None

# src/indexing/indexer.py
from elasticsearch import Elasticsearch

def index_document(es_client, index_name, doc_id, document_text):
    """Indexes a document into Elasticsearch."""
    try:
        es_client.index(index=index_name, id=doc_id, body={"text": document_text})
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False

# src/retrieval/retriever.py
from elasticsearch import Elasticsearch

def search_case_laws(es_client, index_name, query):
    """Searches Elasticsearch for case laws based on a query."""
    try:
        response = es_client.search(index=index_name, body={"query": {"match": {"text": query}}})
        hits = response["hits"]["hits"]
        return [hit["_source"]["text"] for hit in hits]
    except Exception as e:
        print(f"Error searching case laws: {e}")
        return []

# src/qa/qa_engine.py
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import HuggingFaceEmbeddings

def setup_qa_engine(es_client, index_name, model_path):
    """Sets up the question answering engine."""
    embeddings = HuggingFaceEmbeddings()
    vectorstore = ElasticsearchStore(es_client=es_client, index_name=index_name, embedding=embeddings)
    llm = LlamaCpp(model_path=model_path, n_ctx=2048) #Adjust n_ctx to your model's context window.
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa

def answer_question(qa_engine, question):
    """Answers a legal question using the QA engine."""
    try:
        result = qa_engine({"query": question})
        return result["result"]
    except Exception as e:
        print(f"Error answering question: {e}")
        return None

# requirements.txt
streamlit
elasticsearch
langchain
llama-cpp-python
Pillow
pytesseract
sentence-transformers
