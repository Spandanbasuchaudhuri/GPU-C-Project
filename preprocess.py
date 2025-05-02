# preprocess.py
import re
from typing import List
import config

def segment_sentences(text: str) -> List[str]:
    """Split text into sentences using a lightweight regex."""
    parts = re.split(r'(?<=[\.!\?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(
    text: str,
    max_words: int = config.CHUNK_SIZE_WORDS,
) -> List[str]:
    """
    Break the text into chunks of up to max_words each, preserving sentence boundaries.
    """
    sentences = segment_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    wc = 0

    for sent in sentences:
        count = len(sent.split())
        # if adding this sentence would overflow, start a new chunk
        if current and wc + count > max_words:
            chunks.append(" ".join(current))
            current, wc = [], 0

        current.append(sent)
        wc += count

    # append any leftover
    if current:
        chunks.append(" ".join(current))

    return chunks