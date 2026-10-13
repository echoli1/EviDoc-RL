#!/usr/bin/env python3
"""Build the private, page-disjoint EviDoc-RL v1 controlled-GRPO corpus.

The upstream dataset's official ``train`` and ``test`` splits have disjoint
page images at the pinned revision.  This script preserves that boundary:

* all five candidates for a training task come only from official train pages;
* all five candidates for an evaluation task come only from official test pages;
* hard negatives are recomputed independently with split-local OCR BM25;
* only OCR-verifiable, answer-leak-free tasks are selected for GRPO.

It is intentionally a private data preparation script.  The derived page
images, OCR text, and answers must not be committed to the public repository.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rank_bm25 import BM25Okapi

from evidoc_data import (
    DEFAULT_DATASET,
    answer_matches_text,
    extract_english_question,
    file_sha256,
    get_git_commit,
    tokenize,
)


OFFICIAL_SPLITS = ("train", "test")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_sha1(image) -> str:
    """Hash normalized pixels without PNG re-encoding every duplicate row."""
    rgb = image.convert("RGB")
    digest = hashlib.sha1()
    digest.update(f"{rgb.mode}|{rgb.size}".encode("utf-8"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def load_source_rows(dataset_name: str, revision: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required; install requirements.txt first") from exc

    page_by_hash: dict[str, dict] = {}
    records_by_split: dict[str, list[dict]] = {split: [] for split in OFFICIAL_SPLITS}
    source_stats: dict[str, dict] = {}

    for split in OFFICIAL_SPLITS:
        dataset = load_dataset(dataset_name, split=split, revision=revision)
        valid_records = 0
        for source_row, row in enumerate(dataset):
            image = row["image"].convert("RGB")
            page_hash = image_sha1(image)
            page = page_by_hash.get(page_hash)
            if page is None:
                page = {
                    "image": image.copy(),
                    "image_sha1": page_hash,
                    "ocr_text": " ".join(str(word) for word in (row.get("words") or [])),
                    "source_split": split,
                }
                page_by_hash[page_hash] = page
            elif page["source_split"] != split:
                raise RuntimeError(
                    "Official source splits share a page image. Refusing to build a "
                    "page-disjoint benchmark."
                )

            answers = row.get("answers") or ([] if not row.get("answer") else [row["answer"]])
            answers = [str(answer) for answer in answers if str(answer).strip()]
            question = extract_english_question(row.get("query", ""))
            if not question or not answers:
                continue
            valid_records += 1
            records_by_split[split].append(
                {
                    "source_split": split,
                    "source_row": source_row,
                    "question_id": str(row.get("id", source_row)),
                    "question": question,
                    "answers": answers,
                    "image_sha1": page_hash,
                }
            )
        source_stats[split] = {
            "source_rows": len(dataset),
            "valid_english_question_answer_rows": valid_records,
        }

    return page_by_hash, records_by_split, source_stats


def materialize_pages(
    page_by_hash: dict[str, dict], out_dir: Path, dataset_name: str
) -> tuple[list[dict], dict[str, str]]:
    """Assign globally unique, deterministic page ids without split-coded ids."""
    page_dir = out_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict] = []
    page_id_by_hash: dict[str, str] = {}
    for index, page_hash in enumerate(sorted(page_by_hash)):
        page = page_by_hash[page_hash]
        page_id = f"page_{index:05d}"
        image_name = f"{page_id}.png"
        page["image"].save(page_dir / image_name)
        page_id_by_hash[page_hash] = page_id
        pages.append(
            {
                "page_id": page_id,
                "image_path": str(Path("pages") / image_name),
                "ocr_text": page["ocr_text"],
                "source": dataset_name,
                "source_split": page["source_split"],
                "image_sha1": page_hash,
            }
        )
    return pages, page_id_by_hash


def candidate_task(record: dict, page_store: dict[str, dict], page_ids: list[str], bm25, seed: int):
    scores = bm25.get_scores(tokenize(record["question"]))
    page_index = {page_id: index for index, page_id in enumerate(page_ids)}
    gt_page_id = record["gt_page_id"]
    negatives = sorted(
        (
            (page_id, float(scores[index]))
            for index, page_id in enumerate(page_ids)
            if page_id != gt_page_id
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:4]
    if len(negatives) != 4:
        raise RuntimeError(f"{record['question_id']}: fewer than four negative pages")

    gt_matches = answer_matches_text(record["answers"], page_store[gt_page_id]["ocr_text"])
    leaked_answers_by_page = {
        page_id: answer_matches_text(record["answers"], page_store[page_id]["ocr_text"])
        for page_id, _ in negatives
    }
    leaked_answers_by_page = {
        page_id: answers for page_id, answers in leaked_answers_by_page.items() if answers
    }
    if not gt_matches or leaked_answers_by_page:
        return None, {
            "gt_ocr_answer_match": bool(gt_matches),
            "answer_leak_negative": bool(leaked_answers_by_page),
        }

    candidates = [
        {
            "page_id": gt_page_id,
            "is_gt": True,
            "negative_source": None,
            "bm25_score": float(scores[page_index[gt_page_id]]),
        }
    ]
    candidates.extend(
        {
            "page_id": page_id,
            "is_gt": False,
            "negative_source": "bm25_ocr_split_local",
            "bm25_score": score,
        }
        for page_id, score in negatives
    )
    candidate_order_seed = f"{seed}:{record['source_split']}:{record['question_id']}"
    random.Random(candidate_order_seed).shuffle(candidates)
    task = {
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
        "answer_leak_negative": False,
        "leak_negative_page_ids": [],
        "candidate_page_ids": [candidate["page_id"] for candidate in candidates],
        "candidates": candidates,
        "candidate_policy": "1_gt_plus_4_split_local_bm25_ocr_hard_negatives",
        "candidate_order_seed": candidate_order_seed,
    }
    return task, None


def build_clean_tasks(records, pages, page_id_by_hash, split: str, seed: int):
    page_store = {page["page_id"]: page for page in pages}
    split_pages = [page for page in pages if page["source_split"] == split]
    page_ids = [page["page_id"] for page in split_pages]
    if len(page_ids) < 5:
        raise RuntimeError(f"{split}: fewer than five unique pages")
    bm25 = BM25Okapi([tokenize(page["ocr_text"]) for page in split_pages])
    prepared_records = []
    for record in records:
        prepared = dict(record)
        prepared["gt_page_id"] = page_id_by_hash[record["image_sha1"]]
        prepared_records.append(prepared)
    random.Random(f"{seed}:{split}:selection").shuffle(prepared_records)

    clean, filter_stats = [], Counter()
    for record in prepared_records:
        task, rejected = candidate_task(record, page_store, page_ids, bm25, seed)
        if task is None:
            filter_stats["rejected"] += 1
            if not rejected["gt_ocr_answer_match"]:
                filter_stats["gt_ocr_answer_miss"] += 1
            if rejected["answer_leak_negative"]:
                filter_stats["answer_leak_negative"] += 1
            continue
        clean.append(task)
    return clean, {
        "source_records": len(prepared_records),
        "unique_pages": len(split_pages),
        "clean_records": len(clean),
        "rejected_records": filter_stats["rejected"],
        "rejected_gt_ocr_answer_miss": filter_stats["gt_ocr_answer_miss"],
        "rejected_answer_leak_negative": filter_stats["answer_leak_negative"],
    }


def select_tasks(tasks: list[dict], count: int, label: str) -> list[dict]:
    if len(tasks) < count:
        raise RuntimeError(f"Only {len(tasks)} clean {label} tasks are available; requested {count}.")
    selected = []
    for index, task in enumerate(tasks[:count]):
        selected_task = dict(task)
        selected_task["task_id"] = f"v1_{label}_{index:04d}"
        selected.append(selected_task)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/v1")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset_revision", required=True)
    parser.add_argument("--train_tasks", type=int, default=256)
    parser.add_argument("--eval_tasks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_tasks <= 0 or args.eval_tasks <= 0:
        raise ValueError("--train_tasks and --eval_tasks must be positive")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    page_by_hash, records_by_split, source_stats = load_source_rows(args.dataset, args.dataset_revision)
    pages, page_id_by_hash = materialize_pages(page_by_hash, out_dir, args.dataset)
    train_all, train_stats = build_clean_tasks(
        records_by_split["train"], pages, page_id_by_hash, "train", args.seed
    )
    eval_all, eval_stats = build_clean_tasks(
        records_by_split["test"], pages, page_id_by_hash, "test", args.seed
    )
    train_tasks = select_tasks(train_all, args.train_tasks, "train")
    eval_tasks = select_tasks(eval_all, args.eval_tasks, "eval")

    train_candidate_pages = {page_id for task in train_tasks for page_id in task["candidate_page_ids"]}
    eval_candidate_pages = {page_id for task in eval_tasks for page_id in task["candidate_page_ids"]}
    overlap = train_candidate_pages & eval_candidate_pages
    if overlap:
        raise RuntimeError(f"Page-disjoint invariant violated: {sorted(overlap)[:5]}")

    write_jsonl(out_dir / "pages.jsonl", pages)
    write_jsonl(out_dir / "tasks_train.jsonl", train_tasks)
    write_jsonl(out_dir / "tasks_eval.jsonl", eval_tasks)
    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": "evidoc_page_disjoint_v1",
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
            "train_candidate_page_count": len(train_candidate_pages),
            "eval_candidate_page_count": len(eval_candidate_pages),
        },
        "counts": {
            "pages_total": len(pages),
            "pages_by_source_split": dict(Counter(page["source_split"] for page in pages)),
            "selected_train_tasks": len(train_tasks),
            "selected_eval_tasks": len(eval_tasks),
            "clean_pool": {"train": train_stats, "eval": eval_stats},
            "source": source_stats,
        },
        "clean_criteria": {
            "question_language": "en",
            "gt_ocr_answer_match": True,
            "answer_leak_negative": False,
        },
        "candidate_policy": "1_gt_plus_4_split_local_bm25_ocr_hard_negatives",
        "prepare_script_sha256": file_sha256(script_path),
        "git_commit": get_git_commit(script_path.parent),
        "notes": [
            "Private derived data only; do not commit source images, OCR text, questions, or answers.",
            "Document/page-disjointness is verified over every selected candidate page, not only GT pages.",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
