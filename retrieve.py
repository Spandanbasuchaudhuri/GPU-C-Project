"""
retrieve.py  –  query helpers
"""
from __future__ import annotations
from typing import List
import embed_index as ei
import numpy as np, faiss

# ── generic helper ───────────────────────────────────────────────────────
def _ann_search(index: faiss.Index, vec: np.ndarray, top_k: int) -> List[int]:
    k = min(top_k, index.ntotal)
    if k == 0:
        return []
    _, I = index.search(vec, k)    # type: ignore
    return [int(i) for i in I[0]]

# ── text ─────────────────────────────────────────────────────────────────
def retrieve_text(query: str, chunks: List[str], k: int = 5) -> List[str]:
    if ei.TEXT_INDEX is None or ei.TEXT_INDEX.ntotal == 0:
        return []
    vec = ei._encode([query])
    return [chunks[i] for i in _ann_search(ei.TEXT_INDEX, vec, k)]

# ── tables ──────────────────────────────────────────────────────────────
def retrieve_tables(query: str, k: int = 3) -> List[str]:
    out, vec = [], ei._encode([query])
    for idx, rows in ei.TABLE_INDICES.values():
        out.extend(rows[i] for i in _ann_search(idx, vec, k))
    return out

# ── figures ─────────────────────────────────────────────────────────────
def retrieve_figures(query: str, paths: List[str], k: int = 3) -> List[str]:
    if ei.FIG_INDEX.ntotal == 0 or not paths:
        return []
    vec = ei._encode_clip_text(query)
    return [paths[i] for i in _ann_search(ei.FIG_INDEX, vec, k)]

# ── audio ───────────────────────────────────────────────────────────────
def retrieve_audio(query: str, chunks: List[str], k: int = 3) -> List[str]:
    if ei.AUDIO_INDEX.ntotal == 0:
        return []
    vec = ei._encode([query])
    return [chunks[i] for i in _ann_search(ei.AUDIO_INDEX, vec, k)]