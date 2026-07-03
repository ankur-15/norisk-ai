import re

# This regex looks for THREE kinds of boundaries where one clause likely ends
# and another begins:
_SECTION_PATTERN = re.compile(
    r"""
    (?:\n\s*\n)                     # a blank line (paragraph break)
    |(?:\n\s*\d+\.\d*\s+)           # a numbered clause like "4.2 "
    |(?:\n\s*\([a-z]\)\s+)          # a lettered sub-clause like "(a) "
    """,
    re.VERBOSE,
)


def split_into_candidate_sections(text: str) -> list[str]:
    parts = _SECTION_PATTERN.split(text)
    # filter out empty strings / whitespace-only leftovers from the split
    return [p.strip() for p in parts if p and p.strip()]

from transformers import AutoTokenizer

_tokenizer = None  # loaded once, reused — loading it is slow, don't repeat it per chunk


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return _tokenizer


def sliding_window_split(text: str, chunk_size: int = 256, overlap: int = 32) -> list[str]:
    tokenizer = _get_tokenizer()

    # Convert text to token IDs (numbers), not actual sub-word strings yet
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= chunk_size:
        return [text]  # already fits, no splitting needed

    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids))  # convert back to readable text
        if end == len(token_ids):
            break
        start = end - overlap  # step back by `overlap` so we don't cut a sentence mid-thought
    return chunks


def chunk_contract(text: str) -> list[str]:
    sections = split_into_candidate_sections(text)
    chunks = []
    for section in sections:
        if len(section) < 20:  # skip junk like a lone "1." with nothing after it
            continue
        chunks.extend(sliding_window_split(section))
    return chunks