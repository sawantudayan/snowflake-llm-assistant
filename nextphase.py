import json
import logging
import os
import uuid

import faiss
import fitz
import nltk
import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

from utils.embedding_utils import generate_embeddings
from utils.snowflake_connector import get_snowflake_connection

nltk.download('punkt')
load_dotenv()

logger = logging.getLogger(__name__)
HF_API_KEY = os.getenv("HF_API_KEY")


# ----------------------- PDF PROCESSING ----------------------------

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str, max_tokens: int = 512):
    paragraphs = text.split("\n\n")
    chunks, current_chunk = [], ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk.split()) + len(para.split()) <= max_tokens:
            current_chunk += "\n" + para if current_chunk else para
        else:
            if len(para.split()) > max_tokens:
                for sentence in sent_tokenize(para):
                    if len(current_chunk.split()) + len(sentence.split()) <= max_tokens:
                        current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            else:
                chunks.append(current_chunk.strip())
                current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# ---------------------- CLASSIFIER -------------------------------

def is_model_ready():
    health_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    return requests.get(health_url, headers=headers).status_code == 200


def classify_text(text_chunk):
    api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    labels = ["Invoice", "Contract", "Policy", "Technical Spec", "Meeting Notes", "Other"]
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(api_url, headers=headers, json={
        "inputs": text_chunk[:1000],
        "parameters": {"candidate_labels": labels}
    })
    return response.json()["labels"][0] if response.status_code == 200 else "Other"


# ---------------------- SNOWFLAKE I/O -------------------------------

def bulk_upload_chunks_with_embeddings(conn, document_id, file_name, category, chunks, embeddings):
    cursor = conn.cursor()

    # Insert document metadata
    cursor.execute("""
        INSERT INTO genai_assistant.unstructured_data.raw_documents
        (document_id, file_name, upload_timestamp, source_stage, status, category)
        VALUES (%s, %s, CURRENT_TIMESTAMP, %s, %s, %s)
    """, (document_id, file_name, 'streamlit', 'processed', category))

    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = str(uuid.uuid4())
        embedding_json = json.dumps(embedding)

        cursor.execute("""
            MERGE INTO genai_assistant.unstructured_data.document_chunks AS target
            USING (
                SELECT %s AS chunk_id,
                       %s AS document_id,
                       %s AS chunk_text,
                       PARSE_JSON(%s) AS embedding,
                       %s AS chunk_index
            ) AS source
            ON target.chunk_id = source.chunk_id
            WHEN MATCHED THEN UPDATE SET 
                target.embedding = source.embedding,
                target.chunk_index = COALESCE(target.chunk_index, source.chunk_index)
            WHEN NOT MATCHED THEN
                INSERT (chunk_id, document_id, chunk_text, embedding, chunk_index)
                VALUES (source.chunk_id, source.document_id, source.chunk_text, source.embedding, source.chunk_index)
        """, (chunk_id, document_id, chunk_text, embedding_json, idx))

    conn.commit()
    cursor.close()


# ---------------------- STREAMLIT UI -------------------------------

st.set_page_config(layout="centered")
st.title("NextPhase.ai - GenAI PDF Uploader")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading and chunking document"):
        text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(text)
        first_chunk = chunks[0] if chunks else "N/A"
        st.info(f"Document split into {len(chunks)} chunks.")

    with st.spinner("Classifying document"):
        category = classify_text(first_chunk) if is_model_ready() else "Other"
        st.success(f"Classified as: **{category}**")

    with st.spinner("Generating embeddings..."):
        embeddings = generate_embeddings(chunks)

    with st.spinner("Uploading all data to Snowflake..."):
        document_id = str(uuid.uuid4())
        conn = get_snowflake_connection()
        bulk_upload_chunks_with_embeddings(conn, document_id, uploaded_file.name, category, chunks, embeddings)
        conn.close()

    with st.spinner("Building FAISS index..."):
        embedding_matrix = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        faiss.write_index(index, "faiss_index.bin")

        with open("chunk_ids.txt", "w") as f:
            for i in range(len(chunks)):
                f.write(f"{i}\n")

    st.success("All steps completed successfully!")
    st.markdown("---")
    st.write("**FAISS Index saved:** `faiss_index.bin`")
    st.write("**Chunk count:**", len(chunks))
