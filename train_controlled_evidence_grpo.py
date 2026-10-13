#!/usr/bin/env python3
"""GRPO on answer/evidence selection after a fixed OCR controller.

Scope: this trains a controlled answer policy, not a free tool-use agent.  The
prompt already contains OCR excerpts from all five candidates.  It is useful as
a compact, honest comparison of outcome-only and evidence-aware reward when a
free tool policy has no non-zero-reward trajectories yet.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForMultimodalLM, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

import evidoc_runtime as runtime


JSON_OBJECT = re.compile(r"\{.*?\}", flags=re.DOTALL)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", "")
        return content if isinstance(content, str) else str(content)
    if isinstance(completion, list) and completion:
        return completion_to_text(completion[-1])
    return str(completion)


def parse_completion(completion, valid_page_ids):
    text = completion_to_text(completion).strip().replace("```json", "").replace("```", "").strip()
    for candidate in [text] + JSON_OBJECT.findall(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        answer = payload.get("answer")
        page = payload.get("evidence_page")
        if isinstance(answer, str) and isinstance(page, str) and page in valid_page_ids:
            return answer.strip(), page, True
    return "", "", False


def make_reward(mode: str, evidence_bonus: float):
    if mode not in {"outcome", "evidence"}:
        raise ValueError(f"Unsupported reward mode: {mode}")

    def controlled_reward(completions, answers, gt_page_id, candidate_page_ids, **kwargs):
        rewards = []
        for completion, golds, gt_page, candidate_ids in zip(
            completions, answers, gt_page_id, candidate_page_ids
        ):
            answer, page, parse_valid = parse_completion(completion, candidate_ids)
            correct = parse_valid and runtime.answer_matches(answer, golds)
            evidence_hit = parse_valid and page == gt_page
            reward = 1.0 if correct else 0.0
            if mode == "evidence" and evidence_hit:
                reward += evidence_bonus
            rewards.append(float(reward))
        return rewards

    controlled_reward.__name__ = f"controlled_{mode}_reward"
    return controlled_reward


def resolve_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def make_dataset(rows):
    return Dataset.from_list(
        [
            {
                "prompt": row["prompt"],
                "answers": row["answers"],
                "gt_page_id": row["gt_page_id"],
                "candidate_page_ids": row["candidate_page_ids"],
                "task_id": row["task_id"],
            }
            for row in rows
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data/v2/controlled_evidence_train.jsonl")
    parser.add_argument("--data_manifest", default="data/v2/controlled_evidence_manifest.json")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--reward_mode", choices=["outcome", "evidence"], default="evidence")
    parser.add_argument("--output_dir", default="outputs/evidoc_v2_controlled_evidence_grpo")
    parser.add_argument("--max_steps", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Recorded V1/V2 setting: eight rollouts per prompt for GRPO.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.3,
        help="Sampling temperature used to create within-prompt GRPO comparisons.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evidence_bonus",
        type=float,
        default=0.25,
        help="Additional reward for selecting the GT evidence page in evidence mode.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus-sampling threshold used during GRPO rollout generation.",
    )
    parser.add_argument(
        "--log_completions",
        action="store_true",
        help="Print full prompts and sampled completions to the terminal log.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.num_generations < 2:
        raise ValueError("GRPO requires at least two sampled completions per prompt")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    if args.evidence_bonus < 0:
        raise ValueError("--evidence_bonus must be non-negative")

    data_path = Path(args.data_file)
    rows = read_jsonl(data_path)
    if not rows:
        raise RuntimeError("Controlled-evidence training data is empty")
    for row in rows:
        if len(row.get("candidate_page_ids", [])) != 5 or not row.get("prompt"):
            raise ValueError(f"{row.get('task_id')}: malformed controlled prompt")
    dataset = make_dataset(rows)
    print(f"Controlled GRPO rows: {len(rows)}")
    print(f"Reward mode: {args.reward_mode}")
    if args.dry_run:
        sample = rows[0]
        print(json.dumps({
            "task_id": sample["task_id"],
            "candidate_page_ids": sample["candidate_page_ids"],
            "prompt_characters": len(sample["prompt"][1]["content"]),
            "reference": sample["supervised_reference"],
        }, indent=2))
        return

    device, dtype = resolve_device_and_dtype()
    bf16_ok = dtype == torch.bfloat16
    print(f"Training device: {device}\nModel dtype: {dtype}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    if any(parameter.device.type == "meta" for parameter in model.parameters()):
        raise RuntimeError("Model contains meta-device parameters after loading")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_length=128,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_generations,
        num_iterations=1,
        learning_rate=args.learning_rate,
        epsilon=0.2,
        loss_type="grpo",
        scale_rewards="group",
        beta=0.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False,
        remove_unused_columns=False,
        bf16=bf16_ok,
        fp16=device.type == "mps" or (device.type == "cuda" and not bf16_ok),
        logging_steps=1,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_generations,
        report_to="none",
        save_strategy="no",
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        args=config,
        processing_class=processor,
        train_dataset=dataset,
        reward_funcs=make_reward(args.reward_mode, args.evidence_bonus),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    output_dir = Path(args.output_dir)
    manifest = json.loads(Path(args.data_manifest).read_text(encoding="utf-8"))
    metadata = {
        "schema_version": "evidoc_controlled_evidence_grpo_run",
        "model": args.model,
        "reward_mode": args.reward_mode,
        "data_file": str(data_path.resolve()),
        "data_file_sha256": file_sha256(data_path),
        "source_manifest": manifest,
        "training": {
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "evidence_bonus": args.evidence_bonus,
            "seed": args.seed,
            "log_completions": args.log_completions,
            "lora_rank": peft_config.r,
            "target_modules": list(peft_config.target_modules),
            "device": str(device),
            "dtype": str(dtype),
        },
        "scope": "controlled_answer_and_evidence_policy_not_free_tool_policy",
    }
    (output_dir / "controlled_grpo_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "trainer_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Controlled {args.reward_mode} GRPO adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
