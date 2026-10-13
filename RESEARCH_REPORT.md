# EviDoc-RL V1/V2 research report

## Executive summary

- **Evidence-aware reward creates more GRPO learning signal.** In matched,
  one-seed controlled experiments, the fraction of rollout groups with
  non-zero within-group reward standard deviation rises from 15.6% to 30.1%
  in V1 and from 16.4% to 26.6% in V2.
- **That signal increase does not yet establish a task improvement.** V1 and
  V2 greedy grounded correctness is essentially unchanged under the short
  training budgets. In V2 sampling, Evidence has three more grounded
  trajectories than Base out of 224, but all policies retain 9/28 Grounded
  Success@8.
- **The primary contribution is the diagnostic stack.** The public release
  makes the data contract, deterministic controller, reward implementation,
  and greedy evaluation inspectable. Aggregate private-development diagnostics
  additionally report teacher-forced likelihoods and sampled behavior rather
  than collapsing everything into a single answer-accuracy number.

## Problem

For document QA, answer correctness alone is insufficient: a model may guess
the answer or extract it from a semantically incorrect candidate page. This is
especially visible when a hard negative happens to contain the accepted answer
string. EviDoc-RL requires the final policy output to include both an answer
and the page identifier cited as its evidence.

The compared rewards are:

```text
Outcome-only:     1[accepted answer]
Evidence-aware:   1[accepted answer] + 0.25[GT evidence page]
```

The additional evidence term does not automatically guarantee higher task
accuracy. Its direct purpose is to make grounded and ungrounded outcomes
distinguishable during relative-reward post-training.

## Experimental design

### Controlled scope

V1/V2 use a deliberately controlled setup: a deterministic controller
retrieves and OCRs all five candidates, then the model generates the final
answer/evidence JSON. Thus, V1/V2 measure post-training of final evidence
selection after fixed OCR context—not learned retrieval, OCR, or visual-page-
pixel use.

### V1: clean hard negatives

V1 uses English questions with an OCR-verifiable GT answer. It filters out any
selected negative whose OCR contains an accepted answer. The official source
train/test split is enforced over every selected candidate page, yielding 256
train tasks and 64 held-out evaluation tasks.

### V2: strict answer-leak challenge

V2 retains the same source-split/candidate-page-disjoint design but requires:

- an OCR-verifiable GT page;
- at least one exposed non-GT page whose OCR contains an accepted answer; and
- an answer-leak page with BM25 score at least the GT score.

The strict official test pool has 28 eligible tasks, all used for evaluation;
there are 128 train tasks. This design tests whether answer-only reward can
mistake a spurious supporting page for grounded evidence.

### Matched training and evaluation

Both policies use Qwen3-VL-2B-Instruct with LoRA on attention projections,
GRPO with eight generations per prompt, temperature 1.3, top-p 1.0, and seed
42. V1 trains for 256 updates and V2 for 128. Greedy evaluation measures
answer accuracy, evidence-page hit, and grounded correctness. V2 additionally
uses fixed-seed sampling: 28 tasks × 8 samples per policy, temperature 1.3,
top-p 1.0, and a shared seed schedule. Sampling and teacher-forced diagnostics
were run in the private development environment; this release reports their
aggregate results, not raw task-linked trajectories or analysis scripts.

## Results

### Reward-signal diagnostic

| Version | Outcome-only informative groups | Evidence-aware informative groups | Interpretation |
| --- | ---: | ---: | --- |
| V1 | 40/256 (15.6%) | 77/256 (30.1%) | Evidence-aware reward produces more reward-diverse rollout groups. |
| V2 | 21/128 (16.4%) | 34/128 (26.6%) | The same pattern survives the stricter answer-leak setting. |

An informative group means its sampled rewards are not all identical; only
then can group-relative GRPO form a non-zero relative advantage. It is a
learning-signal metric, not a downstream task metric.

### Greedy held-out behavior

| Version / policy | Answer accuracy | Evidence hit | Grounded correct |
| --- | ---: | ---: | ---: |
| V1 Base | 25/64 (39.1%) | 40/64 (62.5%) | 17/64 (26.6%) |
| V1 Outcome | 26/64 (40.6%) | 40/64 (62.5%) | 18/64 (28.1%) |
| V1 Evidence | 25/64 (39.1%) | 40/64 (62.5%) | 17/64 (26.6%) |
| V2 Base | 6/28 (21.4%) | 12/28 (42.9%) | 6/28 (21.4%) |
| V2 Outcome | 6/28 (21.4%) | 12/28 (42.9%) | 6/28 (21.4%) |
| V2 Evidence | 6/28 (21.4%) | 11/28 (39.3%) | 6/28 (21.4%) |

Neither experiment supports a reliable greedy grounding gain. The V1 one-task
difference for Outcome and the V2 one-task Evidence decline are not robust
comparative results.

### V2 generated stochastic behavior

The following fixed-seed aggregate was generated in the private development
environment. It is included as a descriptive diagnostic; its raw trajectories
and task-linked analysis tooling are intentionally not redistributed.

| Policy | Sampled answer correct | Sampled evidence hit | Sampled grounded correct | Grounded Success@8 | Correct answer on leak page |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 56/224 (25.0%) | 93/224 (41.5%) | 53/224 (23.7%) | 9/28 (32.1%) | 3/224 (1.34%) |
| Outcome | 57/224 (25.4%) | 100/224 (44.6%) | 55/224 (24.6%) | 9/28 (32.1%) | 2/224 (0.89%) |
| Evidence | 58/224 (25.9%) | 98/224 (43.8%) | 56/224 (25.0%) | 9/28 (32.1%) | 2/224 (0.89%) |

Task-block bootstrap intervals for sampled grounded correctness relative to
Base are `[0.00, +2.23]` percentage points for Outcome and `[-0.89, +3.57]`
for Evidence. The bootstrap resamples prompts, not model-training seeds. It
does not support a stable performance claim. The identical Success@8 is the
most useful practical guardrail against over-interpreting the 2–3 trajectory
difference.

## What the diagnostics mean

The experiments support a narrow causal chain:

```text
evidence-aware verifier
        ↓
more reward-diverse rollout groups
        ↓
more opportunities for non-zero GRPO relative advantages
        ↓
small, non-conclusive distributional shifts under one short run
        ↓
no established greedy or Success@8 improvement
```

Teacher-forced likelihood analysis is consistent with this picture but has a
different meaning: it conditions on the canonical answer before scoring page
identifiers. It is not the probability of an entire generated trajectory.
Actual sampled JSON evaluation is therefore required for this distinction. The
recorded V2 aggregate is reported here, while the raw trajectories and private
sampling/teacher-forced analysis code are not part of this public release.

Rare correct-answer/wrong-evidence leak behavior appears in all sampled
policies (Base 3/224, Outcome 2/224, Evidence 2/224). It is an observable
failure mode, but it does **not** show that Outcome GRPO preferentially
exploits answer leaks.

## Supported and unsupported claims

Supported:

- V1/V2 construction validates candidate-page-disjoint source splits and the
  stated clean or answer-leak data contracts.
- Evidence-aware reward increases observed within-group reward variation in
  both matched one-seed runs.
- The V2 sampled evaluation finds a small, fixed-seed policy difference while
  retaining unchanged Grounded Success@8.

Not supported:

- statistically reliable accuracy or grounding improvement;
- end-to-end learned tool-use, retrieval, OCR, or visual-pixel improvement;
- robustness to a held-out document domain;
- a causal claim that Outcome GRPO exploits answer leaks; or
- a comparison to a fully trained PPO/DAPO-style agent baseline.

## Why this is still a useful portfolio project

The project is evidence of research-engineering judgment: it turns a vague
grounding objective into a data contract, implements a deterministic verifier,
isolates a known cold-start confound, distinguishes logits from generated
behavior, and publishes the negative result instead of presenting a tiny
difference as a gain. A research internship reviewer can inspect the released
chain from data construction to controlled training and greedy evaluation, and
can see the aggregate private-development diagnostics with their limitations.

## Next research step

Do not obtain a stronger conclusion by simply extending the same configuration
for more steps. A research-scale follow-up should pre-register a change that
increases useful rollout diversity, run at least three independent training
seeds, and separately evaluate a future end-to-end tool-use setting.
