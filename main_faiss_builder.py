import pickle

import faiss
import numpy as np
from tqdm import tqdm

from utils.embedding_utils import get_gte_small_embedding
from utils.snowflake_connector import get_snowflake_connection

VECTOR_DIM = 384
INDEX_PATH = "vector_index.faiss"
METADATA_PATH = "chunk_metadata.pkl"


def fetch_chunks_from_snowflake():
    conn = get_snowflake_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT chunk_id, chunk_text, document_id FROM genai_assistant.unstructured_data.document_chunks")
    rows = cursor.fetchall()

    chunks = [{"chunk_id": r[0], "chunk_text": r[1], "document_id": r[2]} for r in rows]
    return chunks


def build_faiss_index(chunks):
    index = faiss.IndexFlatL2(VECTOR_DIM)
    metadata = []

    for chunk in tqdm(chunks, desc="Embedding & Indexing"):
        try:
            embedding = get_gte_small_embedding(chunk["chunk_text"])
            embedding = np.array(embedding, dtype=np.float32)

            if embedding.shape != (VECTOR_DIM,):
                print(f"Skipping chunk {chunk['chunk_id']} due to incorrect shape: {embedding.shape}")
                continue

            index.add(np.expand_dims(embedding, axis=0))
            metadata.append(chunk)
        except Exception as e:
            print(f"Embedding failed for chunk {chunk['chunk_id']}: {e}")

    return index, metadata


def save_index(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    print("Fetching chunks from Snowflake...")
    chunks = fetch_chunks_from_snowflake()
    print(f"Fetched {len(chunks)} chunks.")

    print("Building FAISS index...")
    index, metadata = build_faiss_index(chunks)

    print("Saving index and metadata...")
    save_index(index, metadata)

    print("✅ FAISS index and metadata saved.")
