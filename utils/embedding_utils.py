import logging
from typing import List

import requests

# Set up logging
logger = logging.getLogger(__name__)

# Correct Hugging Face API settings
HUGGINGFACE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_API_URL = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"

HF_API_KEY = "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"  # Replace with env var or hardcoded for testing

HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}


def validate_hf_model_connection() -> bool:
    """
    Validates the model endpoint and token.
    """
    logger.info("🔍 Validating Hugging Face API access and model availability...")

    try:
        response = requests.get(HUGGINGFACE_API_URL, headers=HEADERS)
        if response.status_code == 200:
            logger.info("✅ Hugging Face model is accessible.")
            return True
        elif response.status_code == 401:
            logger.error("❌ Unauthorized: Invalid or missing Hugging Face API token.")
        elif response.status_code == 404:
            logger.error("❌ Model not found. Check model name.")
        else:
            logger.error(f"❌ Unexpected error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error while connecting to Hugging Face: {e}")

    return False


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Sends a list of texts to Hugging Face and returns embeddings.
    """
    if not validate_hf_model_connection():
        raise RuntimeError("Failed to connect to Hugging Face model endpoint.")

    logger.info(f"📡 Sending {len(texts)} texts to Hugging Face for embedding...")

    try:
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=HEADERS,
            json={"sentences": texts}  # Updated to use 'sentences' instead of 'inputs'
        )
        response.raise_for_status()

        embeddings = response.json()

        # Validate format
        if not isinstance(embeddings, list):
            raise ValueError("❌ Invalid response format: Expected list of embeddings.")

        logger.info("✅ Embeddings received successfully.")
        return embeddings

    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ Hugging Face API error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"❌ Unexpected error during embedding: {e}")

    return []
