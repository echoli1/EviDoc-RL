#!/usr/bin/env python3
"""Freeze controlled OCR evidence prompts for page-disjoint V1/V2 GRPO.

This is deliberately *not* a free tool-use experiment. A deterministic
controller retrieves and OCRs all five candidate pages, after which a policy
must emit a compact answer plus an evidence-page id. Freezing the prompts makes
the post-training target reproducible for controlled evidence selection.
"""

import argparse
import hashlib
import json
from pathlib import Path

import evidoc_runtime as runtime
from controlled_prompting import (
    build_answer_prompt,
    query_conditioned_excerpt,
    ranked_page_ids,
)


SYSTEM_PROMPT = (
    "You are a precise document QA assistant. Follow the required JSON schema exactly."
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_row(task: dict, max_ocr_chars: int) -> dict:
    env = runtime.EviDocEvidenceEnv()
    env.reset(
        task_id=task["task_id"],
        question=task["question"],
        answers=task["answers"],
        gt_page_id=task["gt_page_id"],
        candidate_page_ids=task["candidate_page_ids"],
    )
    retrieval = env.retrieve_pages(task["question"], top_k=5)
    page_ids = ranked_page_ids(retrieval)
    if len(page_ids) != 5 or set(page_ids) != set(task["candidate_page_ids"]):
        raise RuntimeError(f"{task['task_id']}: controller did not expose all five candidates")
    evidence_packs = []
    for page_id in page_ids:
        ocr_text = env.ocr(page_id)
        evidence_packs.append(
            (page_id, query_conditioned_excerpt(task["question"], ocr_text, max_ocr_chars))
        )
    prompt = build_answer_prompt(task["question"], evidence_packs)
    return {
        "task_id": task["task_id"],
        "question_id": task["question_id"],
        "question": task["question"],
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "answers": task["answers"],
        "gt_page_id": task["gt_page_id"],
        "candidate_page_ids": task["candidate_page_ids"],
        "answer_leak_negative": bool(task.get("answer_leak_negative", False)),
        "leak_negative_page_ids": task.get("leak_negative_page_ids", []),
        "controller": {
            "policy": "retrieve_then_ocr_every_runtime_candidate",
            "retrieval_observation": retrieval,
            "ranked_page_ids": page_ids,
            "max_ocr_chars_per_page": max_ocr_chars,
            "uses_gt_during_control": False,
            "uses_answer_labels_during_control": False,
        },
        "supervised_reference": {
            "answer": task["answers"][0],
            "evidence_page": task["gt_page_id"],
        },
    }


def infer_task_contract(train_tasks: list[dict], eval_tasks: list[dict]) -> dict:
    """Infer the V1-clean or V2-answer-leak contract from prepared tasks."""
    tasks = train_tasks + eval_tasks
    leak_flags = {bool(task.get("answer_leak_negative", False)) for task in tasks}
    if leak_flags == {False}:
        return {
            "name": "v1_clean",
            "answer_leak_negative": False,
            "definition": "No selected non-GT candidate OCR contains an accepted answer.",
        }
    if leak_flags == {True}:
        if any(not task.get("leak_bm25_outranks_or_ties_gt", False) for task in tasks):
            raise RuntimeError("V2 answer-leak tasks must include a leak that ties/outranks GT BM25")
        return {
            "name": "v2_answer_leak",
            "answer_leak_negative": True,
            "definition": (
                "Every task exposes a non-GT OCR page containing an accepted answer; "
                "at least one selected leak ties or outranks GT BM25."
            ),
        }
    raise RuntimeError("Prepared train/eval tasks mix clean and answer-leak contracts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data/v2",
        help="Directory containing pages.jsonl; task and output paths may be overridden separately.",
    )
    parser.add_argument("--train_tasks_file", default="data/v2/tasks_train.jsonl")
    parser.add_argument("--eval_tasks_file", default="data/v2/tasks_eval.jsonl")
    parser.add_argument("--train_file", default="data/v2/controlled_evidence_train.jsonl")
    parser.add_argument("--eval_file", default="data/v2/controlled_evidence_eval.jsonl")
    parser.add_argument("--manifest_file", default="data/v2/controlled_evidence_manifest.json")
    parser.add_argument("--max_ocr_chars", type=int, default=1200)
    args = parser.parse_args()
    if args.max_ocr_chars < 200:
        raise ValueError("--max_ocr_chars must be at least 200")

    data_dir = Path(args.data_dir)
    pages_path = data_dir / "pages.jsonl"
    train_tasks_path = Path(args.train_tasks_file)
    eval_tasks_path = Path(args.eval_tasks_file)
    train_tasks = read_jsonl(train_tasks_path)
    eval_tasks = read_jsonl(eval_tasks_path)
    pages = read_jsonl(pages_path)
    if not train_tasks or not eval_tasks:
        raise ValueError("Prepared train/eval task files must both be non-empty")
    train_candidate_pages = {
        page_id for task in train_tasks for page_id in task["candidate_page_ids"]
    }
    eval_candidate_pages = {
        page_id for task in eval_tasks for page_id in task["candidate_page_ids"]
    }
    overlap = train_candidate_pages & eval_candidate_pages
    if overlap:
        raise RuntimeError(f"Prepared tasks are not page-disjoint: {sorted(overlap)[:5]}")
    runtime.PAGE_STORE = runtime.make_page_store(pages)

    train_rows = [build_row(task, args.max_ocr_chars) for task in train_tasks]
    eval_rows = [build_row(task, args.max_ocr_chars) for task in eval_tasks]
    task_contract = infer_task_contract(train_tasks, eval_tasks)

    train_file, eval_file, manifest_file = map(
        Path, (args.train_file, args.eval_file, args.manifest_file)
    )
    for path in (train_file, eval_file, manifest_file):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_file, train_rows)
    write_jsonl(eval_file, eval_rows)
    manifest = {
        "schema_version": f"evidoc_controlled_evidence_{task_contract['name']}",
        "source_train_tasks": str(train_tasks_path.resolve()),
        "source_train_tasks_sha256": file_sha256(train_tasks_path),
        "source_eval_tasks": str(eval_tasks_path.resolve()),
        "source_eval_tasks_sha256": file_sha256(eval_tasks_path),
        "source_pages_sha256": file_sha256(pages_path),
        "controller_policy": "retrieve_then_ocr_every_runtime_candidate",
        "max_ocr_chars_per_page": args.max_ocr_chars,
        "num_train_tasks": len(train_rows),
        "num_eval_tasks": len(eval_rows),
        "train_task_ids": [row["task_id"] for row in train_rows],
        "eval_task_ids": [row["task_id"] for row in eval_rows],
        "split": {
            "type": "prebuilt_official_page_disjoint_split",
            "candidate_page_overlap": len(overlap),
            "guarantee": "Every selected candidate page belongs to exactly one split.",
        },
        "policy_scope": "answer_and_evidence_selection_after_fixed_ocr_controller",
        "uses_visual_page_pixels": False,
        "task_contract": task_contract,
    }
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Built controlled-evidence data: {len(train_rows)} train / {len(eval_rows)} eval tasks."
    )
    print(f"Train: {train_file}\nEval: {eval_file}\nManifest: {manifest_file}")


if __name__ == "__main__":
    main()
