import logging
import time

import faiss
import numpy as np

from utils.embedding_utils import generate_embeddings, validate_hf_model_connection
from utils.snowflake_utils import fetch_chunks_from_snowflake

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_faiss_index():
    start_time = time.time()
    logger.info("Starting FAISS index build process...")

    try:
        # Validate model readiness
        if not validate_hf_model_connection():
            raise RuntimeError("Embedding model is not available or failed to load.")

        # Fetch chunks from Snowflake
        logger.info("Fetching document chunks from Snowflake...")
        chunks = fetch_chunks_from_snowflake()
        if not chunks:
            raise ValueError("No chunks found in Snowflake.")

        texts = [chunk["chunk_text"] for chunk in chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]

        logger.info(f"{len(texts)} document chunks retrieved. Generating embeddings...")

        # Generate local embeddings (batching handled internally)
        embeddings = generate_embeddings(texts)

        if not embeddings or len(embeddings) != len(texts):
            raise RuntimeError("Mismatch or failure in embedding generation.")

        # Convert to float32 NumPy array for FAISS
        embedding_matrix = np.array(embeddings, dtype=np.float32)

        # Build and train FAISS index (FlatL2 = brute-force L2)
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        # Save FAISS index
        faiss.write_index(index, "faiss_index.bin")
        with open("chunk_ids.txt", "w") as f:
            for chunk_id in chunk_ids:
                f.write(f"{chunk_id}\n")

        logger.info(f"FAISS index built and saved with {len(chunk_ids)} entries.")

    except Exception as e:
        logger.error(f"Error during FAISS index build: {e}")
        raise

    finally:
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"FAISS indexing completed in {elapsed} seconds.")


if __name__ == "__main__":
    build_faiss_index()
