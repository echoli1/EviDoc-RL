"""Deterministic OCR controller used by the public V1/V2 experiment.

The policy does not choose tools in this release.  This module only exposes a
reproducible BM25-then-OCR controller for the five already selected candidates.
"""

import re
from typing import List

from evidoc_data import tokenize
from rank_bm25 import BM25Okapi


PAGE_STORE = {}
MAX_OCR_CHARS = 5000


def make_page_store(pages):
    """Return the OCR/page metadata needed by the controlled text-only policy.

    Materialized page images are part of corpus construction, but the released
    V1/V2 controller never reads pixels.  Keeping image paths as inert metadata
    lets prompt construction run from authorized OCR artifacts alone.
    """
    return {page["page_id"]: dict(page) for page in pages}


def normalize_answer(text: str):
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"^[\s\"']+|[\s\"']+$", "", text)


def answer_matches(prediction: str, golds: List[str]) -> bool:
    """Strict normalized equality used for answer reward and evaluation."""
    normalized_prediction = normalize_answer(prediction)
    return bool(normalized_prediction) and any(
        normalized_prediction == normalize_answer(gold) for gold in golds
    )


class EviDocEvidenceEnv:
    """State holder for deterministic candidate retrieval and OCR access."""

    def __init__(self):
        self.reset()

    def reset(
        self,
        task_id=None,
        question=None,
        answers=None,
        gt_page_id=None,
        candidate_page_ids=None,
        **_unused,
    ):
        self.task_id = task_id
        self.question = question
        self.answers = list(answers or [])
        self.gt_page_id = gt_page_id
        self.candidate_page_ids = list(candidate_page_ids or [])
        self.retrieve_calls = 0
        self.ocr_calls = 0
        self.invalid_calls = 0
        self.action_history = []

    def _valid_page(self, page_id: str) -> bool:
        return page_id in self.candidate_page_ids and page_id in PAGE_STORE

    def retrieve_pages(self, query: str, top_k: int = 5) -> str:
        self.retrieve_calls += 1
        self.action_history.append(
            {"tool": "retrieve_pages", "query": str(query), "top_k": int(top_k)}
        )
        top_k = max(1, min(int(top_k), len(self.candidate_page_ids)))
        docs = [tokenize(PAGE_STORE[page_id]["ocr_text"]) for page_id in self.candidate_page_ids]
        scores = BM25Okapi(docs).get_scores(tokenize(query))
        ranked = sorted(
            zip(self.candidate_page_ids, scores), key=lambda item: float(item[1]), reverse=True
        )[:top_k]
        return "\n".join(
            f"{rank + 1}. {page_id} score={float(score):.4f}"
            for rank, (page_id, score) in enumerate(ranked)
        )

    def ocr(self, page_id: str) -> str:
        self.action_history.append({"tool": "ocr", "page_id": str(page_id)})
        if not self._valid_page(page_id):
            self.invalid_calls += 1
            return f"Invalid page_id: {page_id}"
        self.ocr_calls += 1
        text = PAGE_STORE[page_id]["ocr_text"]
        if len(text) > MAX_OCR_CHARS:
            return text[:MAX_OCR_CHARS] + "\n[OCR TRUNCATED]"
        return text
