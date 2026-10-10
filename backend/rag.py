"""
Retrieval-augmented Q&A over openFDA label sections.

Embeddings: a small neural model turns a piece of text into a list of
384 numbers (a vector) such that texts with similar MEANING end up close
together, even with different words ("keep refrigerated" vs. "store at
2-8C"). Retrieval = embed the question, then return the chunks whose
vectors are closest to it (cosine similarity, 1.0 = same direction).

Sections can be thousands of words, and one vector for a whole section
blurs its meaning, so sections are split into ~120-word chunks first.
all-MiniLM-L6-v2 is ~90MB, runs fine on CPU, and reads at most 256
tokens per input, which 120 words stays under.
"""

import hashlib
import json
import os
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

from extract import _call_groq_with_retry

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 120

_model = SentenceTransformer(MODEL_NAME)  # loaded once at import, like the OCR reader
_index_cache = {}  # set_id -> (chunks, embeddings)


def chunk_sections(sections: dict) -> list[dict]:
    """Split each section into sentence-aligned chunks of <= CHUNK_WORDS
    words, keeping the section name on every chunk."""
    chunks = []
    for section, text in sections.items():
        sentences = re.split(r"(?<=[.!?])\s+", text)
        current = []
        for sentence in sentences:
            words = sentence.split()
            if current and len(current) + len(words) > CHUNK_WORDS:
                chunks.append({"section": section, "text": " ".join(current)})
                current = []
            current.extend(words)
        if current:
            chunks.append({"section": section, "text": " ".join(current)})
    return chunks


def _index(label: dict):
    key = label.get("set_id")
    if key in _index_cache:
        return _index_cache[key]
    chunks = chunk_sections(label["sections"])
    # Prefix the section name so e.g. "storage and handling" is part of
    # what the chunk "means", even if its text never says "storage".
    texts = [f"{c['section'].replace('_', ' ')}: {c['text']}" for c in chunks]
    embeddings = _model.encode(texts, normalize_embeddings=True)
    _index_cache[key] = (chunks, embeddings)
    return chunks, embeddings


def retrieve(question: str, label: dict, k: int = 4) -> list[dict]:
    """Top-k chunks of the label most similar to the question."""
    chunks, embeddings = _index(label)
    if not chunks:
        return []
    query = _model.encode([question], normalize_embeddings=True)[0]
    # vectors are normalized, so a dot product IS the cosine similarity
    scores = embeddings @ query
    top = scores.argsort()[::-1][:k]
    return [{**chunks[i], "score": round(float(scores[i]), 3)} for i in top]


# ---------- answer generation (the "G" in RAG) ----------
#
# The LLM sees ONLY the top-k retrieved chunks, numbered [1]..[k], and must
# cite them; anything not in them is "not found". Q&A runs on its own model
# (QA_MODEL) because Groq quotas are per model, so label Q&A can't eat into
# the extraction model's daily tokens.

QA_MODEL = os.getenv("QA_MODEL", "openai/gpt-oss-120b")
NOT_FOUND = "Not found in the official FDA label."
QA_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "cache" / "qa_cache.json"

QA_PROMPT = """You answer questions about a medicine using ONLY the numbered \
excerpts from its official FDA label given below.

Rules:
- Use only facts stated in the excerpts. No outside knowledge, no guessing.
- Cite every fact with the excerpt number(s) in square brackets, e.g. [2].
- If the excerpts give different instructions for different presentations \
or states (e.g. vial vs. pen vs. cartridge, opened vs. unopened, in-use vs. \
not in-use), answer separately for each one and never merge them into a \
single rule.
- Keep the answer short: 1-4 sentences per presentation, plain language.
- If the excerpts do not contain the answer, set "found" to false.

Return ONLY a JSON object: {"found": true/false, "answer": "...", "citations": [numbers used]}"""

_QA_PROMPT_HASH = hashlib.sha256(QA_PROMPT.encode("utf-8")).hexdigest()[:16]


def _load_qa_cache() -> dict:
    if QA_CACHE_PATH.exists():
        return json.loads(QA_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _answer_key(question: str, label: dict, k: int) -> str:
    return hashlib.sha256(f"{QA_MODEL}:{_QA_PROMPT_HASH}:{label.get('set_id')}:{k}:"
                          f"{question.strip().lower()}".encode("utf-8")).hexdigest()


def is_answer_cached(question: str, label: dict, k: int = 4) -> bool:
    """Would answer() hit the cache? (lets the eval print a Groq budget)"""
    return _answer_key(question, label, k) in _load_qa_cache()


def answer(question: str, label: dict, k: int = 4) -> dict:
    """Answer a question from the label's top-k chunks, with citations.

    Returns {"found": bool, "answer": str, "citations": [{"n", "section", "text", "score"}]}
    """
    question = question.strip()
    key = _answer_key(question, label, k)
    cache = _load_qa_cache()
    if key in cache:
        return cache[key]

    chunks = retrieve(question, label, k=k)
    excerpts = "\n\n".join(f"[{i}] ({c['section'].replace('_', ' ')}) {c['text']}"
                           for i, c in enumerate(chunks, start=1))
    response = _call_groq_with_retry(
        model=QA_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": QA_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
        ],
    )
    try:
        data = json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        data = {"found": False}

    cited = [n for n in data.get("citations") or [] if isinstance(n, int) and 1 <= n <= len(chunks)]
    if not data.get("found") or not data.get("answer") or not cited:
        result = {"found": False, "answer": NOT_FOUND, "citations": []}
    else:
        result = {
            "found": True,
            "answer": data["answer"].strip(),
            "citations": [{"n": n, **chunks[n - 1]} for n in sorted(set(cited))],
        }

    cache[key] = result
    QA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QA_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return result
