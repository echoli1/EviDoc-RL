# EviDoc-RL

**Evidence-grounded post-training for document agents.**

Document QA agents can produce the correct answer while grounding that answer
in the wrong page. EviDoc-RL studies this failure mode by requiring a policy to
jointly predict an `answer` and its supporting `evidence_page` among
hard-negative document candidates.

The project compares **outcome-only GRPO**, which rewards final-answer
correctness, with **evidence-aware GRPO**, which additionally rewards correct
evidence grounding. Rather than evaluating only final accuracy, EviDoc-RL
traces the optimization chain:

```text
reward variation
→ group-relative advantage
→ GRPO update opportunities
→ policy-distribution shift
→ held-out grounded performance
```

The main empirical finding is consistent across two controlled experiments:
**evidence-aware rewards produce substantially denser GRPO learning signals,
but the additional signal does not yield a reliable held-out greedy-performance
improvement at the current scale.**

## Experimental scope

This public release contains the **controlled evidence-policy** experiment.
A deterministic controller retrieves and OCRs five candidate pages;
Qwen3-VL-2B-Instruct then predicts only the final JSON `answer` and
`evidence_page`. This isolates answer generation and evidence selection from
tool-protocol failures.

The V1/V2 results therefore should not be interpreted as end-to-end learned
tool use, learned retrieval/OCR, or improved end-to-end visual reasoning.

Both experiments use page-disjoint train/evaluation splits: candidate pages
used for evaluation are excluded from task-specific post-training.

## V1 and V2

**V1 — clean hard-negative setting.** Each task has one ground-truth evidence
page and four OCR/BM25 hard negatives. Selected negative pages are filtered so
they do not contain an accepted answer.

**V2 — answer-leak challenge.** Each task includes at least one non-GT page
that contains an accepted answer. For every V2 task, at least one selected leak
page lexically ties or outranks the GT page under BM25, creating a targeted
setting where answer correctness alone may fail to identify proper grounding.

## Main findings

| Experiment | Outcome-only informative GRPO groups | Evidence-aware informative GRPO groups | Held-out greedy grounded correctness |
| --- | ---: | ---: | --- |
| V1 clean, 256 train / 64 eval | 40/256 (15.6%) | **77/256 (30.1%)** | Base 17/64; Outcome 18/64; Evidence 17/64 |
| V2 answer-leak, 128 train / 28 eval | 21/128 (16.4%) | **34/128 (26.6%)** | Base 6/28; Outcome 6/28; Evidence 6/28 |

Across both settings, evidence-aware reward increases the fraction of rollout
groups with non-zero within-group reward variation. This creates more
opportunities for GRPO to form non-zero relative advantages; it does not by
itself establish a non-zero parameter update for every group. However, that
denser signal does **not** translate into a reliable improvement in held-out
greedy answer accuracy, evidence selection, or grounded correctness in these
experiments.

A stochastic V2 evaluation shows a small descriptive shift toward more
grounded sampled outputs under Evidence GRPO (`+1.34` percentage points versus
Base), but its task-block bootstrap interval crosses zero. We therefore treat
it as a policy-shift diagnostic rather than a performance result.

Read the full evidence and interpretation in
[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md). The machine-readable aggregate is
[`reports/v1_v2_aggregate_results.json`](reports/v1_v2_aggregate_results.json).

> **Research status — code-only release.** The reported V1/V2 experiments use
> one training seed. They do not establish a statistically reliable accuracy
> gain, held-out-domain robustness, or an end-to-end tool-use improvement.

## Quick start — no data or GPU

```bash
python examples/run_synthetic_demo.py
```

The synthetic demo is completely fictional. It matches the published V1/V2
reward definitions: citing the GT evidence page earns an additional reward
only in the evidence-aware condition.

## Reproduce with independently authorized data

No DocVQA-derived task rows, images, OCR text, questions, answers, adapters,
or detailed trajectories are distributed. Obtain upstream data independently
and review its terms before running the construction scripts. All derived data
remain under `data/`, which Git ignores.

The recorded experiments use `nielsr/docvqa_1200_examples` revision
`dc77ab0c3d98855d0f3cb3a00832f2423fbe7528`. The following command reconstructs
the V1 data contract locally; it writes private derived artifacts under
`data/v1/`, which is ignored by Git:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python prepare_v1_page_disjoint.py \
  --out_dir data/v1 \
  --dataset_revision dc77ab0c3d98855d0f3cb3a00832f2423fbe7528 \
  --train_tasks 256 --eval_tasks 64 --seed 42
python validate_v1_page_disjoint.py --data_dir data/v1
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for V1/V2 prompt construction,
matched GRPO configurations, greedy evaluation, and what a fresh run can—and
cannot—be compared against.

## Repository map

| Area | Files |
| --- | --- |
| Public result narrative | `RESEARCH_REPORT.md`, `reports/v1_v2_aggregate_results.json` |
| V1 clean data contract | `prepare_v1_page_disjoint.py`, `validate_v1_page_disjoint.py` |
| V2 answer-leak contract | `prepare_v2_answer_leak.py`, `validate_v2_answer_leak.py` |
| Deterministic OCR controller and prompt | `evidoc_runtime.py`, `controlled_prompting.py`, `prepare_controlled_evidence.py` |
| GRPO / greedy evaluation | `train_controlled_evidence_grpo.py`, `evaluate_controlled_evidence.py` |
| Shared corpus utilities | `evidoc_data.py` |

## Citation and scope

This is an engineering/research portfolio project, not a peer-reviewed paper.
If you build on it, cite the repository commit and preserve the distinction
between controlled evidence selection and end-to-end tool use.

Licensed under the [MIT License](LICENSE).
