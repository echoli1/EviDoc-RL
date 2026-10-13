"""Prompt construction for answer-and-evidence selection after fixed OCR."""

import re

from evidoc_data import tokenize


RANK_LINE = re.compile(r"^\d+\.\s+(page_[^\s]+)\s+score=", flags=re.MULTILINE)


def ranked_page_ids(retrieval_observation: str):
    page_ids = RANK_LINE.findall(retrieval_observation)
    if not page_ids:
        raise ValueError(f"Could not parse controller retrieval output: {retrieval_observation!r}")
    return page_ids


def query_conditioned_excerpt(question: str, ocr_text: str, max_chars: int):
    """Deterministically compact OCR without inspecting any answer labels."""
    if len(ocr_text) <= max_chars:
        return ocr_text
    query_tokens = set(tokenize(question))
    words = ocr_text.split()
    windows = []
    for start in range(0, len(words), 40):
        window = words[start : start + 80]
        if window:
            overlap = sum(token in query_tokens for token in tokenize(" ".join(window)))
            windows.append((overlap, -start, " ".join(window)))
    windows.sort(reverse=True)

    selected, used = [], 0
    for _, _, window in windows:
        if window in selected:
            continue
        separator = "\n...\n" if selected else ""
        remaining = max_chars - used - len(separator)
        if remaining <= 0:
            break
        selected.append(window[:remaining])
        used += len(separator) + len(selected[-1])
    return "\n...\n".join(selected)[:max_chars]


def build_answer_prompt(question: str, evidence_packs):
    evidence_text = "\n\n".join(
        f"[Evidence page: {page_id}]\n{excerpt}" for page_id, excerpt in evidence_packs
    )
    return (
        "Answer the document question using only the OCR evidence below. "
        "Return exactly one JSON object and no Markdown: "
        '{"answer": "short answer", "evidence_page": "page_id"}. '
        "The answer must be the shortest verbatim answer span copied from the OCR. "
        "Do not add a full sentence, explanation, subject, punctuation, or any words that are not needed to answer the question. "
        "The evidence_page must be one of the supplied page ids. If the evidence is insufficient, "
        'return {"answer": "", "evidence_page": ""}.\n\n'
        f"Question: {question}\n\n{evidence_text}"
    )
