import faiss
import numpy as np

def build_faiss_index(embeddings: list, dim: int = 384):
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index