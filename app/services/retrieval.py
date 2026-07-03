import faiss
import numpy as np
from app.services.embedding import embed_texts

EMBEDDING_DIM = 384  # must match the model's output size


class ClauseIndex:
    def __init__(self):
        # IndexFlatIP = "Inner Product" search = cosine similarity, since our
        # vectors are already normalized (dot product == cosine similarity)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.texts = []  # keep original text aligned with each vector's position

    def add(self, texts: list[str]):
        vectors = embed_texts(texts)
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query: str, top_k: int = 3):
        query_vector = embed_texts([query])
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.texts[idx], float(score)))
        return results