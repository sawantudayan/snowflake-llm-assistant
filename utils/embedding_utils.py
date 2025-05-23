import logging
import os
from typing import List

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

from utils.snowflake_utils import insert_embeddings_to_snowflake

# Load env vars
load_dotenv()
logger = logging.getLogger(__name__)
print(torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "thenlper/gte-small")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, force_download=True)
    model = AutoModel.from_pretrained(MODEL_NAME, force_download=True).to(DEVICE).eval()
    logger.info(f"Successfully loaded {MODEL_NAME} on {DEVICE}")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    raise


def mean_pooling(last_hidden_state, attention_mask):
    """Performs mean pooling on token embeddings."""
    expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    return (last_hidden_state * expanded_mask).sum(1) / expanded_mask.sum(1)


def validate_hf_model_connection() -> bool:
    """
    Simulates validation for local model. Always returns True unless model failed to load.
    """
    try:
        _ = tokenizer("test", return_tensors="pt")
        logger.info("Local embedding model ready.")
        return True
    except Exception as e:
        logger.error(f"Local model validation failed: {e}")
        return False


def generate_embeddings(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using a local Hugging Face model.
    """
    if not validate_hf_model_connection():
        raise RuntimeError("Local Hugging Face embedding model not available.")

    logger.info(f"Generating embeddings locally for {len(texts)} texts...")

    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        pooled = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        pooled_np = pooled.cpu().numpy()

        embeddings.extend(pooled_np.tolist())

    logger.info("Embeddings generated successfully.")
    return embeddings


def generate_embeddings_and_store_in_snowflake(texts: List[str], chunk_ids: List[str], document_ids: List[str],
                                               batch_size: int = 16):
    """
    Generates embeddings for a list of texts and stores them in Snowflake.
    """
    embeddings = generate_embeddings(texts, batch_size)

    for i, embedding in enumerate(embeddings):
        chunk_id = chunk_ids[i]  # Assuming you have chunk_id, document_id lists
        document_id = document_ids[i]
        chunk_text = texts[i]

        # Insert the embedding and other data into Snowflake
        insert_embeddings_to_snowflake(chunk_id, document_id, chunk_text, embedding)

    print("Embeddings inserted into Snowflake successfully.")
