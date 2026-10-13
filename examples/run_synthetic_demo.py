#!/usr/bin/env python3
"""Run a dependency-free, fully fictional EviDoc-RL reward demonstration."""

import json
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def reward(answer_correct: bool, evidence_correct: bool) -> tuple[float, float]:
    """Return the published V1/V2 outcome and evidence-aware rewards."""
    outcome_only = float(answer_correct)
    evidence_aware = outcome_only + (0.25 if evidence_correct else 0.0)
    return outcome_only, evidence_aware


def main() -> None:
    task_path = Path(__file__).with_name("synthetic_task_example.json")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    prediction = task["demo_prediction"]
    answer_correct = normalize(prediction["answer"]) == normalize(task["gold_answer"])
    evidence_correct = prediction["evidence_page"] == task["gold_evidence_page"]
    outcome, evidence = reward(answer_correct, evidence_correct)
    wrong_outcome, wrong_evidence = reward(answer_correct, False)

    print("EviDoc-RL synthetic reward demo")
    print(f"Question: {task['question']}")
    print(f"Prediction: {prediction['answer']} @ {prediction['evidence_page']}")
    print(f"Answer correct: {answer_correct}")
    print(f"Evidence correct: {evidence_correct}")
    print(f"Outcome-only reward: {outcome:.2f}")
    print(f"Evidence-aware reward: {evidence:.2f}")
    print()
    print("Counterfactual: same correct answer, but cite page_a")
    print(f"Outcome-only reward: {wrong_outcome:.2f}")
    print(f"Evidence-aware reward: {wrong_evidence:.2f}")


if __name__ == "__main__":
    main()
