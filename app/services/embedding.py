from sentence_transformers import SentenceTransformer

_model = None  # loaded once and reused, same pattern as the tokenizer


def _get_model():
    global _model
    if _model is None:
        # all-MiniLM-L6-v2: small, fast, good general-purpose sentence embedding model
        # outputs 384-dimensional vectors
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]):
    model = _get_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # scales each vector to length 1
    )
    return embeddings