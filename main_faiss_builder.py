import logging
import time

import faiss
import numpy as np

from utils.embedding_utils import generate_embeddings
from utils.snowflake_utils import fetch_chunks_from_snowflake

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_faiss_index():
    start_time = time.time()
    logger.info("🔄 Fetching document chunks from Snowflake...")

    try:
        # Fetch document chunks from Snowflake
        chunks = fetch_chunks_from_snowflake()
        if not chunks:
            raise ValueError("No chunks found in Snowflake.")

        # Extract texts and chunk IDs
        texts = [chunk['chunk_text'] for chunk in chunks]
        chunk_ids = [chunk['chunk_id'] for chunk in chunks]

        # Generate embeddings
        logger.info("🧠 Generating embeddings using Hugging Face API...")
        embeddings = generate_embeddings(texts)

        if not embeddings:
            raise RuntimeError("No embeddings generated. FAISS index cannot be built.")

        # Convert to float32 NumPy array for FAISS
        embedding_matrix = np.array(embeddings).astype("float32")

        # Build FAISS index
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        # Optional: Save the FAISS index and metadata
        faiss.write_index(index, "faiss_index.bin")
        with open("chunk_ids.txt", "w") as f:
            for chunk_id in chunk_ids:
                f.write(f"{chunk_id}\n")

        logger.info(f"✅ FAISS index built and saved with {len(chunk_ids)} entries.")

    except Exception as e:
        logger.error(f"❌ Error during FAISS index build: {e}")
        raise e

    finally:
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"⏱️ Total time: {elapsed} seconds.")


if __name__ == "__main__":
    build_faiss_index()
