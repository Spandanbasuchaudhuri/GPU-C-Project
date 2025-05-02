"""
app.py  –  Streamlit frontend
"""
from pathlib import Path
import streamlit as st, tempfile, uuid, time, logging, types, importlib, atexit

# ── patch torch before Streamlit's watcher touches it ────────────────────
def _patch_torch_classes() -> None:
    try:
        m = importlib.import_module("torch._classes")
        if not hasattr(m, "__path__"):
            m.__path__ = types.SimpleNamespace(_path=[])
    except Exception as e:
        logging.warning(f"torch patch failed: {e}")
_patch_torch_classes()

import config, embed_index as ei
from ingest     import extract_text_pdf, extract_text_pptx, ocr_image, transcribe_audio
from preprocess import chunk_text
import retrieve as R
from generate   import build_prompt, call_ollama_stream

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Streamlit page ───────────────────────────────────────────────────────
st.set_page_config("📄 Document Assistant", layout="wide")
st.title("📄 Document Assistant")
st.caption("Upload PDFs / PPTX / images / audio → ask questions!")

MAX_BYTES  = config.MAX_FILE_SIZE_MB * 1024 * 1024
UPLOAD_DIR = Path(tempfile.gettempdir()) / f"docassist_{uuid.uuid4().hex}"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── sidebar – uploads ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Uploads")
    pdf   = st.file_uploader("PDF",  type=["pdf"])
    pptx  = st.file_uploader("PPTX", type=["pptx"])
    imgs  = st.file_uploader("Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    audio = st.file_uploader("Audio",  type=["wav", "mp3"])
    go    = st.button("Process")

def _save(up) -> str | None:
    if up and len(up.getbuffer()) > MAX_BYTES:
        st.warning(f"{up.name} > {config.MAX_FILE_SIZE_MB} MB ⟹ skipped")
        return None
    p = UPLOAD_DIR / f"{int(time.time()*1e6)}_{up.name}"
    p.write_bytes(up.getbuffer())
    return str(p)

if go:
    if not any([pdf, pptx, imgs, audio]):
        st.error("Upload at least one file."); st.stop()

    t0 = time.perf_counter()
    with st.status("⏳ Extracting & indexing…", expanded=True) as status:
        ei.clear_all()                 # robust reset ✔
        text, fig_paths, aud_chunks = "", [], []

        # PDF
        if pdf:
            status.write(f"PDF → {pdf.name}")
            if (p := _save(pdf)):
                text += extract_text_pdf(p)

        # PPTX
        if pptx:
            status.write(f"PPTX → {pptx.name}")
            if (p := _save(pptx)):
                text += "\n" + extract_text_pptx(p)

        # images
        for im in imgs or []:
            status.write(f"OCR → {im.name}")
            if (p := _save(im)):
                ocr = ocr_image(p)
                if ocr:
                    text += "\n[IMG]\n" + ocr
                    fig_paths.append(p)

        # audio
        if audio:
            status.write(f"ASR → {audio.name}")
            if (p := _save(audio)):
                tr = transcribe_audio(p)
                size = config.AUDIO_CHUNK_SIZE
                aud_chunks = [tr[i:i+size] for i in range(0, len(tr), size)]

        if not any([text.strip(), fig_paths, aud_chunks]):
            status.update(state="error", label="Nothing readable found"); st.stop()

        # chunk + index
        ch = chunk_text(text)
        ei.index_text_chunks(ch)
        ei.index_table_rows(ch)
        ei.index_figures(fig_paths)
        ei.index_audio_chunks(aud_chunks)

        status.update(state="complete", label=f"✅ Ready in {time.perf_counter()-t0:.1f}s")

    # store in session
    st.session_state.update({
        "chunks": ch, "figs": fig_paths, "aud": aud_chunks,
        "chat": [], "processed": True,
    })
    st.success("Done! Ask away…")

# ── chat history ─────────────────────────────────────────────────────────
hist  = st.session_state.get("chat", [])
for role, msg in hist:
    st.chat_message(role).markdown(msg)

q = st.chat_input("Ask anything…")
if q:
    if not st.session_state.get("processed"):
        st.error("Process docs first."); st.stop()

    st.session_state.chat.append(("user", q))
    st.chat_message("user").markdown(q)

    txt = R.retrieve_text   (q, st.session_state.chunks, config.TOP_K_TEXT)
    tbl = R.retrieve_tables (q, config.TOP_K_TABLE)
    fig = R.retrieve_figures(q, st.session_state.figs, config.TOP_K_FIG)
    aud = R.retrieve_audio  (q, st.session_state.aud, config.TOP_K_AUDIO)

    prompt  = build_prompt(txt, tbl, fig, aud, st.session_state.chat, q)
    box     = st.chat_message("assistant").empty()
    answer  = ""

    for tok in call_ollama_stream(prompt):
        answer += tok; box.markdown(answer)
    st.session_state.chat.append(("assistant", answer))

# ── housekeeping ─────────────────────────────────────────────────────────
if st.button("🧹 Clear chat"): st.session_state.chat = []; st.rerun()
@atexit.register
def _cleanup(): import shutil, os; shutil.rmtree(UPLOAD_DIR, ignore_errors=True)