from app.services.chunking import chunk_contract
from app.services.retrieval import ClauseIndex

def build_index_from_contract(text: str) -> ClauseIndex:
    chunks = chunk_contract(text)
    index = ClauseIndex()
    index.add(chunks)
    return index