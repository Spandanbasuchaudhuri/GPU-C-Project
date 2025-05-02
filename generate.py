"""
generate.py  –  prompt assembly + Ollama streaming
"""
import logging, time
from typing import List, Tuple, Generator
import config, textwrap

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def build_prompt(
    text:  List[str], tables: List[str], figs: List[str], audio: List[str],
    history: List[Tuple[str, str]], question: str,
) -> str:
    # last 6 turns, nicely indented
    hist = "\n".join(
        f"User: {u}\nAssistant: {a}" for u, a in history[-6:]
        if u and a
    )
    def _block(title: str, items: List[str]) -> str:
        if not items:
            return f"{title}:\n  (none)"
        joined = "\n".join(f"- {i}" for i in items)
        return f"{title}:\n{joined}"

    return textwrap.dedent(f"""\
        == Conversation ==
        {hist}

        {_block("Text",   text)}
        {_block("Tables", tables)}
        {_block("Figures", figs)}
        {_block("Audio",  audio)}

        Question: {question}
        Answer (be concise, cite brackets where appropriate):
    """)

def call_ollama_stream(prompt: str, model: str = config.OLLAMA_MODEL) -> Generator[str,None,None]:
    try:
        from ollama import chat
    except ImportError:
        yield "ERROR: `pip install ollama` first."; return

    msgs = [
        {"role": "system", "content": "You are a helpful document assistant."},
        {"role": "user",   "content": prompt},
    ]
    for chunk in chat(model=model, messages=msgs, stream=True):
        # both possible shapes → unify
        if "message" in chunk:
            yield chunk["message"]["content"]
        elif "content" in chunk:
            yield chunk["content"]