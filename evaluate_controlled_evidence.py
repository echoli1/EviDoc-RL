#!/usr/bin/env python3
"""Evaluate a frozen-controller answer/evidence policy on held-out prompts."""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

import evidoc_runtime as runtime
from train_controlled_evidence_grpo import parse_completion, read_jsonl


def resolve_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def load_policy(model_name: str, adapter_dir: str | None):
    device, dtype = resolve_device_and_dtype()
    print(f"Evaluation device: {device}\nModel dtype: {dtype}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.to(device)
    model.eval()
    return model, processor, device


@torch.inference_mode()
def generate(model, processor, device, prompt, max_new_tokens: int):
    rendered = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], return_tensors="pt", padding=True).to(device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    completion_ids = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(completion_ids, skip_special_tokens=True)[0]


def summarize(traces):
    total = len(traces)
    correct_wrong_evidence = [
        trace for trace in traces if trace["answer_correct"] and not trace["evidence_page_hit"]
    ]
    return {
        "num_tasks": total,
        "json_parse_valid_rate": sum(trace["parse_valid"] for trace in traces) / total,
        "answer_accuracy": sum(trace["answer_correct"] for trace in traces) / total,
        "evidence_page_hit": sum(trace["evidence_page_hit"] for trace in traces) / total,
        "grounded_correct_rate": sum(
            trace["answer_correct"] and trace["evidence_page_hit"] for trace in traces
        ) / total,
        "answer_correct_wrong_evidence_rate": len(correct_wrong_evidence) / total,
        "answer_correct_wrong_evidence_count": len(correct_wrong_evidence),
        "answer_correct_leak_evidence_rate": sum(
            trace["answer_correct"] and trace["submitted_evidence_page"] in trace["leak_negative_page_ids"]
            for trace in traces
        ) / total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data/v2/controlled_evidence_eval.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter_dir", help="Omit for the untuned base policy.")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, help="Optional prefix of the held-out set for a smoke test.")
    parser.add_argument("--output_file", default="reports/controlled_evidence_eval.json")
    args = parser.parse_args()
    rows = read_jsonl(Path(args.data_file))
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("Evaluation data is empty")
    model, processor, device = load_policy(args.model, args.adapter_dir)
    traces = []
    for index, row in enumerate(rows, start=1):
        completion = generate(model, processor, device, row["prompt"], args.max_new_tokens)
        answer, evidence_page, parse_valid = parse_completion(completion, row["candidate_page_ids"])
        answer_correct = parse_valid and runtime.answer_matches(answer, row["answers"])
        evidence_hit = parse_valid and evidence_page == row["gt_page_id"]
        trace = {
            "task_id": row["task_id"],
            "answers": row["answers"],
            "gt_page_id": row["gt_page_id"],
            "leak_negative_page_ids": row.get("leak_negative_page_ids", []),
            "raw_completion": completion,
            "parse_valid": parse_valid,
            "submitted_answer": answer,
            "submitted_evidence_page": evidence_page,
            "answer_correct": answer_correct,
            "evidence_page_hit": evidence_hit,
        }
        traces.append(trace)
        print(
            f"[{index}/{len(rows)}] {row['task_id']} "
            f"answer={answer_correct} evidence={evidence_hit} parse={parse_valid}"
        )
    report = {
        "schema_version": "evidoc_controlled_evidence_eval",
        "mode": "frozen_ocr_controller_answer_evidence_policy",
        "model": args.model,
        "adapter_dir": args.adapter_dir,
        "source_data": str(Path(args.data_file).resolve()),
        "summary": summarize(traces),
        "traces": traces,
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
