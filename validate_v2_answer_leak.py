#!/usr/bin/env python3
"""Fail fast if the private V2 answer-leak challenge violates its contract."""

import argparse
import json
from pathlib import Path

from evidoc_data import answer_matches_text


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_tasks(tasks, page_store, expected_split):
    seen_pages, errors = set(), []
    for task in tasks:
        task_id = task.get("task_id", "<missing-task-id>")
        candidates = task.get("candidate_page_ids", [])
        gt_page_id = task.get("gt_page_id")
        if task.get("source_split") != expected_split:
            errors.append(f"{task_id}: unexpected source_split")
        if len(candidates) != 5 or len(set(candidates)) != 5 or gt_page_id not in candidates:
            errors.append(f"{task_id}: malformed candidate set")
            continue
        if any(page_id not in page_store for page_id in candidates):
            errors.append(f"{task_id}: unknown candidate page")
            continue
        if any(page_store[page_id]["source_split"] != expected_split for page_id in candidates):
            errors.append(f"{task_id}: cross-split candidate page")
        answers = task.get("answers", [])
        if not answer_matches_text(answers, page_store[gt_page_id]["ocr_text"]):
            errors.append(f"{task_id}: GT OCR does not contain an accepted answer")
        actual_leaks = {
            page_id
            for page_id in candidates
            if page_id != gt_page_id and answer_matches_text(answers, page_store[page_id]["ocr_text"])
        }
        saved_leaks = set(task.get("leak_negative_page_ids", []))
        if not actual_leaks:
            errors.append(f"{task_id}: no answer-leak negative")
        if saved_leaks != actual_leaks:
            errors.append(f"{task_id}: saved leak ids do not match OCR")
        scores = {item["page_id"]: item["bm25_score"] for item in task.get("candidates", [])}
        if not any(
            scores.get(page_id, float("-inf")) >= scores.get(gt_page_id, float("inf"))
            for page_id in actual_leaks
        ):
            errors.append(f"{task_id}: no answer-leak candidate outranks/ties GT BM25")
        seen_pages.update(candidates)
    return seen_pages, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/v2")
    parser.add_argument("--report_file", default="reports/v2_data_validation.json")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    pages = read_jsonl(data_dir / "pages.jsonl")
    page_store = {page["page_id"]: page for page in pages}
    train_tasks = read_jsonl(data_dir / "tasks_train.jsonl")
    eval_tasks = read_jsonl(data_dir / "tasks_eval.jsonl")
    train_pages, train_errors = validate_tasks(train_tasks, page_store, "train")
    eval_pages, eval_errors = validate_tasks(eval_tasks, page_store, "test")
    overlap = train_pages & eval_pages
    errors = train_errors + eval_errors
    if overlap:
        errors.append(f"train/eval candidate page overlap: {sorted(overlap)[:10]}")
    report = {
        "schema_version": "evidoc_answer_leak_challenge_v2_validation",
        "data_dir": str(data_dir.resolve()),
        "counts": {
            "pages": len(pages),
            "train_tasks": len(train_tasks),
            "eval_tasks": len(eval_tasks),
            "train_candidate_pages": len(train_pages),
            "eval_candidate_pages": len(eval_pages),
            "candidate_page_overlap": len(overlap),
        },
        "valid": not errors,
        "errors": errors,
    }
    output = Path(args.report_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit("V2 data validation failed")


if __name__ == "__main__":
    main()
