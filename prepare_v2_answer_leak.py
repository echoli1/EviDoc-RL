#!/usr/bin/env python3
"""Build a page-disjoint EviDoc-RL V2 answer-leak challenge corpus.

Each selected task has one OCR-verifiable GT page and four split-local BM25
hard negatives. At least one negative also contains an accepted answer and has
a BM25 score no lower than the GT's. The latter makes lexical retrieval alone
an especially plausible route to the wrong evidence.

The corpus is private: it materializes source images, OCR, questions and
answers only under the caller-supplied data directory.
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi

from evidoc_data import DEFAULT_DATASET, answer_matches_text, file_sha256, get_git_commit, tokenize
from prepare_v1_page_disjoint import (
    OFFICIAL_SPLITS,
    load_source_rows,
    materialize_pages,
    write_jsonl,
)


def build_challenge_task(record, page_store, page_ids, bm25, seed):
    """Return one 1-GT + 4-negative task, or a transparent rejection reason."""
    scores = bm25.get_scores(tokenize(record["question"]))
    score_by_id = {page_id: float(scores[index]) for index, page_id in enumerate(page_ids)}
    gt_page_id = record["gt_page_id"]
    gt_matches = answer_matches_text(record["answers"], page_store[gt_page_id]["ocr_text"])
    if not gt_matches:
        return None, "gt_ocr_answer_miss"

    negatives = sorted(
        ((page_id, score) for page_id, score in score_by_id.items() if page_id != gt_page_id),
        key=lambda item: item[1],
        reverse=True,
    )
    leak_candidates = []
    for page_id, score in negatives:
        matches = answer_matches_text(record["answers"], page_store[page_id]["ocr_text"])
        if matches:
            leak_candidates.append((page_id, score, matches))
    if not leak_candidates:
        return None, "no_answer_leak_negative"

    eligible_leaks = [item for item in leak_candidates if item[1] >= score_by_id[gt_page_id]]
    if not eligible_leaks:
        return None, "leak_bm25_below_gt"

    # The highest-scoring qualifying leak must be visible to the policy.  Fill
    # the remaining slots with the strongest other lexical hard negatives.
    selected_leak = eligible_leaks[0]
    selected_negative_ids = [selected_leak[0]]
    for page_id, _ in negatives:
        if page_id != selected_leak[0]:
            selected_negative_ids.append(page_id)
        if len(selected_negative_ids) == 4:
            break
    if len(selected_negative_ids) != 4:
        raise RuntimeError(f"{record['question_id']}: fewer than four negative pages")

    leak_matches = {
        page_id: answer_matches_text(record["answers"], page_store[page_id]["ocr_text"])
        for page_id in selected_negative_ids
    }
    leak_matches = {page_id: matches for page_id, matches in leak_matches.items() if matches}
    candidates = [
        {
            "page_id": gt_page_id,
            "is_gt": True,
            "negative_source": None,
            "bm25_score": score_by_id[gt_page_id],
        }
    ]
    for page_id in selected_negative_ids:
        candidates.append(
            {
                "page_id": page_id,
                "is_gt": False,
                "negative_source": (
                    "bm25_ocr_answer_leak" if page_id in leak_matches else "bm25_ocr_hard_negative"
                ),
                "bm25_score": score_by_id[page_id],
            }
        )
    candidate_order_seed = f"{seed}:{record['source_split']}:{record['question_id']}"
    random.Random(candidate_order_seed).shuffle(candidates)
    leak_page_ids = [page_id for page_id in selected_negative_ids if page_id in leak_matches]
    return {
        "seed": seed,
        "source_split": record["source_split"],
        "source_row": record["source_row"],
        "question_id": record["question_id"],
        "question": record["question"],
        "question_language": "en",
        "answers": record["answers"],
        "gt_page_id": gt_page_id,
        "gt_ocr_answer_match": True,
        "matching_gt_answers": gt_matches,
        "answer_leak_negative": True,
        "leak_negative_page_ids": leak_page_ids,
        "leak_negative_matching_answers": leak_matches,
        "selected_leak_page_id": selected_leak[0],
        "selected_leak_bm25_score": selected_leak[1],
        "gt_bm25_score": score_by_id[gt_page_id],
        "leak_bm25_outranks_or_ties_gt": selected_leak[1] >= score_by_id[gt_page_id],
        "candidate_page_ids": [candidate["page_id"] for candidate in candidates],
        "candidates": candidates,
        "candidate_policy": "1_gt_plus_1_answer_leak_plus_3_split_local_bm25_ocr_hard_negatives",
        "candidate_order_seed": candidate_order_seed,
    }, None


def build_pool(records, pages, page_id_by_hash, split, seed):
    page_store = {page["page_id"]: page for page in pages}
    split_pages = [page for page in pages if page["source_split"] == split]
    page_ids = [page["page_id"] for page in split_pages]
    if len(page_ids) < 5:
        raise RuntimeError(f"{split}: fewer than five unique pages")
    bm25 = BM25Okapi([tokenize(page["ocr_text"]) for page in split_pages])
    prepared = []
    for record in records:
        item = dict(record)
        item["gt_page_id"] = page_id_by_hash[record["image_sha1"]]
        prepared.append(item)
    random.Random(f"{seed}:{split}:v2-challenge-selection").shuffle(prepared)
    accepted, rejections = [], Counter()
    for record in prepared:
        task, reason = build_challenge_task(record, page_store, page_ids, bm25, seed)
        if task is None:
            rejections[reason] += 1
        else:
            accepted.append(task)
    return accepted, {
        "source_records": len(prepared),
        "unique_pages": len(split_pages),
        "challenge_records": len(accepted),
        "rejections": dict(rejections),
    }


def select_tasks(tasks, count, label):
    if len(tasks) < count:
        raise RuntimeError(
            f"Only {len(tasks)} {label} tasks meet the answer-leak policy; requested {count}."
        )
    selected = []
    for index, task in enumerate(tasks[:count]):
        item = dict(task)
        item["task_id"] = f"v2_{label}_{index:04d}"
        selected.append(item)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/v2")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_revision", required=True)
    parser.add_argument("--train_tasks", type=int, default=128)
    parser.add_argument("--eval_tasks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_tasks <= 0 or args.eval_tasks <= 0:
        raise ValueError("--train_tasks and --eval_tasks must be positive")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    page_by_hash, records_by_split, source_stats = load_source_rows(args.dataset, args.dataset_revision)
    pages, page_id_by_hash = materialize_pages(page_by_hash, out_dir, args.dataset)
    train_pool, train_stats = build_pool(
        records_by_split["train"], pages, page_id_by_hash, "train", args.seed
    )
    eval_pool, eval_stats = build_pool(
        records_by_split["test"], pages, page_id_by_hash, "test", args.seed
    )
    train_tasks = select_tasks(train_pool, args.train_tasks, "train")
    eval_tasks = select_tasks(eval_pool, args.eval_tasks, "eval")
    train_pages = {page_id for task in train_tasks for page_id in task["candidate_page_ids"]}
    eval_pages = {page_id for task in eval_tasks for page_id in task["candidate_page_ids"]}
    overlap = train_pages & eval_pages
    if overlap:
        raise RuntimeError(f"Page-disjoint invariant violated: {sorted(overlap)[:5]}")

    write_jsonl(out_dir / "pages.jsonl", pages)
    write_jsonl(out_dir / "tasks_train.jsonl", train_tasks)
    write_jsonl(out_dir / "tasks_eval.jsonl", eval_tasks)
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "evidoc_page_disjoint_answer_leak_challenge_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "seed": args.seed,
        "source_splits": {"train": "train", "eval": "test"},
        "split": {
            "type": "official_split_page_disjoint",
            "group_key": "image_sha1",
            "candidate_corpus": "split_local",
            "train_eval_candidate_page_overlap": len(overlap),
        },
        "counts": {
            "pages_total": len(pages),
            "pages_by_source_split": dict(Counter(page["source_split"] for page in pages)),
            "selected_train_tasks": len(train_tasks),
            "selected_eval_tasks": len(eval_tasks),
            "challenge_pool": {"train": train_stats, "eval": eval_stats},
            "source": source_stats,
        },
        "challenge_criteria": {
            "question_language": "en",
            "gt_ocr_answer_match": True,
            "answer_leak_negative": True,
            "leak_policy": "outrank_gt",
            "leak_definition": "At least one exposed non-GT candidate OCR contains an accepted answer.",
        },
        "candidate_policy": "1_gt_plus_1_answer_leak_plus_3_split_local_bm25_ocr_hard_negatives",
        "prepare_script_sha256": file_sha256(script_path),
        "git_commit": get_git_commit(script_path.parent),
        "notes": [
            "Private derived data only; do not commit source images, OCR text, questions, or answers.",
            "The challenge isolates outcome correctness from evidence correctness: a correct answer may be supported by a non-GT page.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
