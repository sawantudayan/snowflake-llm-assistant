import logging
import time

import faiss
import numpy as np

from utils.embedding_utils import generate_embeddings, validate_hf_model_connection
from utils.snowflake_utils import fetch_chunks_from_snowflake, insert_embeddings_to_snowflake

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_faiss_index_and_store_embeddings():
    start_time = time.time()
    logger.info("Starting FAISS index build process...")

    try:
        if not validate_hf_model_connection():
            raise RuntimeError("Embedding model is not available or failed to load.")

        chunks = fetch_chunks_from_snowflake()
        if not chunks:
            raise ValueError("No chunks found in Snowflake.")

        texts = [chunk["chunk_text"] for chunk in chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        document_ids = [chunk["document_id"] for chunk in chunks]

        logger.info(f"{len(texts)} document chunks retrieved. Generating embeddings...")

        embeddings = generate_embeddings(texts)
        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError("Mismatch or failure in embedding generation.")

        embedding_matrix = np.array(embeddings, dtype=np.float32)

        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        faiss.write_index(index, "faiss_index.bin")
        with open("chunk_ids.txt", "w") as f:
            for chunk_id in chunk_ids:
                f.write(f"{chunk_id}\n")

        logger.info(f"FAISS index built and saved with {len(chunk_ids)} entries.")

        for chunk_id, document_id, chunk_text, embedding in zip(chunk_ids, document_ids, texts, embeddings):
            logger.debug(f"Inserting: {chunk_id}, doc_id={document_id}, emb[0:5]={embedding[:5]}")
            insert_embeddings_to_snowflake(chunk_id, document_id, chunk_text, embedding)

    except Exception as e:
        logger.error(f"Error during FAISS index build and embedding insertion: {e}")
        raise

    finally:
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"FAISS indexing completed in {elapsed} seconds.")


if __name__ == "__main__":
    build_faiss_index_and_store_embeddings()
