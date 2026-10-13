"""Small, dependency-light helpers shared by the V1/V2 corpus builders.

This public release deliberately keeps raw DocVQA-derived artifacts out of
version control.  These functions make the data construction contract explicit
without bundling the private corpus itself.
"""

import ast
import hashlib
import re
import subprocess
from pathlib import Path


DEFAULT_DATASET = "nielsr/docvqa_1200_examples"


def tokenize(text: str):
    """Tokenize text consistently for the split-local BM25 index."""
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def answer_matches_text(answers, text: str):
    """Return accepted answers that occur verbatim in an OCR string."""
    normalized_page = normalize_text(text)
    matches = []
    for answer in answers:
        normalized_answer = normalize_text(answer)
        if normalized_answer and normalized_answer in normalized_page:
            matches.append(str(answer))
    return matches


def extract_english_question(raw_query):
    """Extract DocVQA's English query, including legacy stringified dicts."""
    query = raw_query
    if isinstance(query, str):
        stripped = query.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, dict):
                query = parsed
        if isinstance(query, str):
            return stripped or None
    if isinstance(query, dict):
        english = str(query.get("en", "")).strip()
        return english or None
    return None


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(repo_dir: Path):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
