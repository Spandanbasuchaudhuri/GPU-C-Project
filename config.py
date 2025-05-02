# config.py ───────── central settings ────────────────────────────────────

# ── Models ──
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim
CLIP_MODEL      = "openai/clip-vit-base-patch32"  # 512-dim
WHISPER_MODEL   = "base"
OLLAMA_MODEL    = "gemma:2b"           # or llama3:8b, mistral-7b, …

# ── Hardware ──
CUDA_DEVICE     = 0                    # -1 for CPU only
USE_GPU         = CUDA_DEVICE >= 0

# ── FAISS ──
FAISS_TEXT_DIM  = 384
FAISS_FIG_DIM   = 512

# ── Retrieval ──
TOP_K_TEXT   = 5
TOP_K_TABLE  = 3
TOP_K_FIG    = 3
TOP_K_AUDIO  = 3

# ── Chunking ──
CHUNK_SIZE_WORDS    = 300
CHUNK_OVERLAP_WORDS = 50

# ── Ingestion limits ──
MAX_FILE_SIZE_MB = 50
TESSERACT_LANG   = "eng"
AUDIO_CHUNK_SIZE = 500