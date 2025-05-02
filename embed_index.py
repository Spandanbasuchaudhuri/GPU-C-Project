"""
embed_index.py  –  build & manage FAISS indexes
"""
from __future__ import annotations
import logging, re
from typing import List, Dict, Tuple, Optional

import faiss, numpy as np, torch, pandas as pd
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

import config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── device ───────────────────────────────────────────────────────────────
_device = (
    torch.device(f"cuda:{config.CUDA_DEVICE}")
    if config.USE_GPU and torch.cuda.is_available()
    else torch.device("cpu")
)
log.info(f"Using device: {_device}")

# ── models ───────────────────────────────────────────────────────────────
_text_model  = SentenceTransformer(config.EMBEDDING_MODEL, device=str(_device))
_clip_model  = CLIPModel.from_pretrained(config.CLIP_MODEL).to(_device)
_clip_proc   = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
log.info("Sentence-Transformer and CLIP loaded")

# ── FAISS helpers (GPU optional) ─────────────────────────────────────────
_res, _gpu_cfg = None, None
if config.USE_GPU and faiss.get_num_gpus() > 0:
    _res = faiss.StandardGpuResources()
    _gpu_cfg = faiss.GpuIndexFlatConfig()
    _gpu_cfg.device = config.CUDA_DEVICE
    log.info("FAISS GPU resources ready")

def _new_index(dim: int) -> faiss.Index:
    if _res:
        return faiss.GpuIndexFlatIP(_res, dim, _gpu_cfg)   # GPU
    return faiss.IndexFlatIP(dim)                          # CPU

# ── global indexes ───────────────────────────────────────────────────────
TEXT_INDEX   : Optional[faiss.Index]            = None
FIG_INDEX    : faiss.Index                      = _new_index(config.FAISS_FIG_DIM)
AUDIO_INDEX  : faiss.Index                      = _new_index(config.FAISS_TEXT_DIM)
TABLE_INDICES: Dict[str, Tuple[faiss.Index, List[str]]] = {}

# ── encoding ─────────────────────────────────────────────────────────────
def _encode(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, config.FAISS_TEXT_DIM), np.float32)
    v = _text_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return v.astype(np.float32)

def _encode_clip_text(txt: str) -> np.ndarray:
    batch = _clip_proc(text=[txt], return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        v = _clip_model.get_text_features(**batch)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy().astype(np.float32)

# ── public API ───────────────────────────────────────────────────────────
def clear_all() -> None:
    global TEXT_INDEX, FIG_INDEX, AUDIO_INDEX, TABLE_INDICES
    TEXT_INDEX = None
    FIG_INDEX.reset()
    AUDIO_INDEX.reset()
    TABLE_INDICES.clear()

def index_text_chunks(chunks: List[str]) -> None:
    if not chunks:
        return
    vecs = _encode(chunks)
    if vecs.size == 0:
        return
    global TEXT_INDEX
    TEXT_INDEX = _new_index(vecs.shape[1])  # always fresh to avoid dim-mismatch
    TEXT_INDEX.add(vecs)
    log.info(f"Text index ↳ {TEXT_INDEX.ntotal} vectors")

def index_table_rows(chunks: List[str]) -> None:
    rows = [
        ln.strip()
        for ch in chunks
        for ln in ch.splitlines()
        if ln.strip() and (ln.count("|") >= 2 or "\t" in ln)
    ]
    if not rows:
        return
    vecs = _encode(rows)
    idx  = _new_index(vecs.shape[1])
    idx.add(vecs)
    TABLE_INDICES["__rows__"] = (idx, rows)
    log.info(f"Inline rows ↳ {len(rows)}")

def index_table(df: pd.DataFrame, table_id: str) -> None:
    rows = df.astype(str).agg(" | ".join, axis=1).tolist()
    vecs = _encode(rows)
    idx  = _new_index(vecs.shape[1])
    idx.add(vecs)
    TABLE_INDICES[table_id] = (idx, rows)
    log.info(f"DataFrame {table_id} indexed")

def index_figures(paths: List[str]) -> List[str]:
    good, imgs = [], []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            imgs.append(im); good.append(p)
        except Exception as e:
            log.warning(f"{p}: {e}")

    if not imgs:
        return []

    batch = _clip_proc(images=imgs, return_tensors="pt").to(_device)
    with torch.no_grad():
        feats = _clip_model.get_image_features(**batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    FIG_INDEX.add(feats.cpu().numpy().astype(np.float32))
    log.info(f"Figure index ↳ {FIG_INDEX.ntotal} vectors")
    return good

def index_audio_chunks(chunks: List[str]) -> None:
    vecs = _encode(chunks)
    if vecs.size == 0:
        return
    AUDIO_INDEX.add(vecs)
    log.info(f"Audio index ↳ {AUDIO_INDEX.ntotal} vectors")

# expose everything retrieve.py needs
__all__ = [
    "_encode", "_encode_clip_text", "_device",
    "TEXT_INDEX", "FIG_INDEX", "AUDIO_INDEX", "TABLE_INDICES", "clear_all",
]