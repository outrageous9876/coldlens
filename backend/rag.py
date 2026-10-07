"""
Retrieval over openFDA label sections (the "R" in RAG; no generation yet).

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

import re

from sentence_transformers import SentenceTransformer

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
