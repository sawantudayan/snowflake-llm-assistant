# semantic_search.py
import logging
import os
import pickle

import faiss
import numpy as np

from utils.embedding_utils import generate_embeddings

# Load FAISS index and chunk metadata
logger = logging.getLogger(__name__)

INDEX_PATH = "faiss_index.bin"
CHUNK_ID_PATH = "chunk_ids.txt"
METADATA_PATH = "chunk_metadata.pkl"

logger.info("Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)

with open(CHUNK_ID_PATH, "r") as f:
    chunk_ids = [line.strip() for line in f.readlines()]

with open(METADATA_PATH, "rb") as f:
    chunk_metadata = pickle.load(f)  # {chunk_id: {filename, category, chunk_text}}


def search_semantic(query: str, k: int = 5, filters: dict = None):
    # Load FAISS index and metadata
    faiss_index = faiss.read_index("faiss_index.bin")

    with open("chunk_ids.txt", "r") as f:
        chunk_ids = [line.strip() for line in f.readlines()]

    with open("chunk_metadata.pkl", "rb") as f:
        chunk_metadata = pickle.load(f)  # This is a list of dictionaries

    # Get relevant metadata filters
    if filters:
        filtered_metadata = [
            m for m in chunk_metadata if (
                    (filters.get('filename') and m['filename'] in filters['filename']) or
                    (filters.get('category') and m['category'] in filters['category'])
            )
        ]
    else:
        filtered_metadata = chunk_metadata

    # Process the query (you may want to embed the query as well)
    query_embedding = generate_embeddings([query])[0]  # Assuming single query input

    # Search the FAISS index
    k_nearest = faiss_index.search(np.array([query_embedding], dtype=np.float32), k)

    # Process the results
    results = []
    for idx, dist in zip(k_nearest[1][0], k_nearest[0][0]):
        chunk_id = chunk_ids[idx]

        # Find the corresponding metadata for this chunk_id
        metadata = next((m for m in filtered_metadata if m['chunk_id'] == chunk_id), {})

        results.append({
            'chunk_id': chunk_id,
            'chunk_text': metadata.get('chunk_text', ''),
            'filename': metadata.get('filename', ''),
            'category': metadata.get('category', ''),
            'distance': dist
        })

    return results


def get_all_metadata_filters():
    if not os.path.exists("chunk_metadata.pkl"):
        return [], []

    with open("chunk_metadata.pkl", "rb") as f:
        chunk_metadata = pickle.load(f)  # This is a list of dicts

    filenames = sorted({m['filename'] for m in chunk_metadata})
    categories = sorted({m['category'] for m in chunk_metadata})
    return filenames, categories
