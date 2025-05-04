import os
import uuid
from typing import List

import fitz
import requests
import streamlit as st
from dotenv import load_dotenv

from utils.snowflake_connector import get_snowflake_connection

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Snowflake connection setup
conn = get_snowflake_connection()


# Text extraction using PyMuPDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    return text


# Chunk the text
def chunk_text(text: str, max_tokens: int = 512) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current.split()) + len(para.split()) < max_tokens:
            current += "\n" + para
        else:
            chunks.append(current.strip())
            current = para
    if current:
        chunks.append(current.strip())
    return chunks


# Classify text using Hugging Face - BART LARGE
# TEMPORARY fallback until Cortex is working
def classify_text(text_chunk):
    labels = ["Invoice", "Contract", "Policy", "Technical Spec", "Meeting Notes", "Other"]
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

    payload = {
        "inputs": text_chunk[:1000],  # Trim long text for safety
        "parameters": {"candidate_labels": labels}
    }

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result["labels"][0]
    else:
        raise RuntimeError(f"HF API error: {response.status_code} - {response.text}")


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
    st.info("Extracting text and classifying...")

    # Extract and process
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    first_chunk = chunks[0] if chunks else "No text found."

    # Connect and store
    conn = get_snowflake_connection()
    document_id = str(uuid.uuid4())
    category = classify_text(first_chunk)
    insert_document(conn, document_id, uploaded_file.name, category)
    insert_chunks(conn, document_id, chunks)

    st.success(f"Uploaded and classified as **{category}**")
    st.write(f"Classified as:", category)
    st.write(f"Total chunks stored: {len(chunks)}")
