import os

import requests
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")


def get_gte_small_embedding(text):
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/thenlper/gte-small"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }

    payload = {
        "inputs": text,
        "options": {
            "wait_for_model": True
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API error: {response.status_code} - {response.text}")

    embedding = response.json()
    # The API returns [1, 384] shape → flatten to list
    return embedding[0]
