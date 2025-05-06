import os
import uuid
from typing import List

import fitz
import nltk
import requests
import streamlit as st
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

from utils.snowflake_connector import get_snowflake_connection

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Download sentence tokenizer (only once)
nltk.download('punkt')

# Snowflake connection setup
conn = get_snowflake_connection()


# Text extraction using PyMuPDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text


def chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk.split()) + len(para.split()) <= max_tokens:
            current_chunk += "\n" + para if current_chunk else para
        else:
            if len(para.split()) > max_tokens:
                # Fallback: break long para into sentence-based chunks
                sentences = sent_tokenize(para)
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk.split()) + len(sentence.split()) <= max_tokens:
                        temp_chunk += " " + sentence if temp_chunk else sentence
                    else:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def is_model_ready():
    health_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.get(health_url, headers=headers)
    return response.status_code == 200


# Classify text using Hugging Face - BART LARGE
def classify_text(text_chunk):
    labels = ["Invoice", "Contract", "Policy", "Technical Spec", "Meeting Notes", "Other"]
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

    payload = {
        "inputs": text_chunk[:1000],  # Trim for token limits
        "parameters": {"candidate_labels": labels}
    }

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result["labels"][0]
    else:
        st.error(f"HF API error: {response.status_code} - {response.text}")
        return "Other"  # Fallback category


# Insert into Snowflake
def insert_document(conn, document_id, file_name, category):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO genai_assistant.unstructured_data.raw_documents 
        (document_id, file_name, upload_timestamp, source_stage, status, category)
        VALUES (%s, %s, CURRENT_TIMESTAMP, %s, %s, %s)
    """, (document_id, file_name, 'streamlit', 'processed', category))


def insert_chunks(conn, document_id, chunks):
    cursor = conn.cursor()
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO genai_assistant.unstructured_data.document_chunks 
            (chunk_id, document_id, chunk_index, chunk_text, created_at)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        """, (chunk_id, document_id, i, chunk))


# Streamlit UI
st.title("Snowflake LLM Assistant")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_file is not None:
    st.info("Extracting and processing...")

    # Step 1: Extract text
    text = extract_text_from_pdf(uploaded_file)

    # Step 2: Smart chunking
    chunks = chunk_text(text)
    first_chunk = chunks[0] if chunks else "No text found."

    # Step 3: Classify document
    conn = get_snowflake_connection()
    document_id = str(uuid.uuid4())

    if is_model_ready():
        with st.spinner("Classifying document type using Hugging Face model..."):
            category = classify_text(first_chunk)
    else:
        st.warning("Hugging Face model is not ready. Defaulting to 'Other'.")
        category = "Other"

    # Step 4: Upload to Snowflake
    insert_document(conn, document_id, uploaded_file.name, category)
    insert_chunks(conn, document_id, chunks)

    # Step 5: UI Feedback
    st.success(f"Uploaded and classified as **{category}**")
    st.write("Classification:", category)
    st.write("Total chunks stored:", len(chunks))
