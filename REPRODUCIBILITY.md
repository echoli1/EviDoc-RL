# Reproducibility guide

## Environment

The recorded runs used Python 3.12, PyTorch 2.7.1, Transformers 5.17.0, TRL
1.12.0, PEFT 0.20.0, Qwen3-VL-2B-Instruct, CUDA BF16, and seed 42 for
training. CPU can run corpus validators and the synthetic reward demo, but an
allocated GPU is strongly preferred for model inference and training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reconstruct private V1 data locally

Only after independently obtaining authorized upstream data:

```bash
REVISION=dc77ab0c3d98855d0f3cb3a00832f2423fbe7528
python prepare_v1_page_disjoint.py \
  --out_dir data/v1 --dataset_revision "$REVISION" \
  --train_tasks 256 --eval_tasks 64 --seed 42
python validate_v1_page_disjoint.py --data_dir data/v1

python prepare_controlled_evidence.py \
  --data_dir data/v1 \
  --train_tasks_file data/v1/tasks_train.jsonl \
  --eval_tasks_file data/v1/tasks_eval.jsonl \
  --train_file data/v1/controlled_evidence_train.jsonl \
  --eval_file data/v1/controlled_evidence_eval.jsonl \
  --manifest_file data/v1/controlled_evidence_manifest.json
```

The validator checks five unique candidates, GT OCR verification, no negative
answer leakage, source-split membership, and zero selected candidate-page
overlap between train and evaluation.

## Reconstruct private V2 data locally

The strict official test pool had 28 eligible tasks in the recorded source
revision; therefore use `--eval_tasks 28`, not the script default of 40.

```bash
REVISION=dc77ab0c3d98855d0f3cb3a00832f2423fbe7528
python prepare_v2_answer_leak.py \
  --out_dir data/v2 --dataset_revision "$REVISION" \
  --train_tasks 128 --eval_tasks 28 --seed 42
python validate_v2_answer_leak.py --data_dir data/v2
# This final command uses prepare_controlled_evidence.py's data/v2 defaults.
python prepare_controlled_evidence.py
```

The V2 validator additionally checks that a non-GT candidate has an accepted
answer in OCR and that the saved leak annotation matches the OCR-derived one.

## Matched GRPO runs and greedy evaluation

Run Base evaluation and each reward condition separately on an explicitly
allocated GPU. The following V2 commands show the recorded hyperparameters;
substitute the V1 paths and `--max_steps 256` for V1.

```bash
# Outcome-only GRPO; run a second time with --reward_mode evidence.
CUDA_VISIBLE_DEVICES=<allocated_gpu> python train_controlled_evidence_grpo.py \
  --data_file data/v2/controlled_evidence_train.jsonl \
  --data_manifest data/v2/controlled_evidence_manifest.json \
  --reward_mode outcome --max_steps 128 --num_generations 8 \
  --temperature 1.3 --top_p 1.0 --learning_rate 5e-6 --seed 42 \
  --output_dir outputs/v2_outcome

# Omit --adapter_dir for the Base policy.
CUDA_VISIBLE_DEVICES=<allocated_gpu> python evaluate_controlled_evidence.py \
  --data_file data/v2/controlled_evidence_eval.jsonl \
  --adapter_dir outputs/v2_outcome \
  --output_file reports/v2_outcome_greedy.json
```

The aggregate report records the published greedy and private-development
sampling diagnostics. Raw sampled trajectories, teacher-forced analysis code,
and task-linked diagnostic reports are intentionally not public because they
contain upstream-derived task content. Fresh runs can vary because model/
download/runtime versions and upstream data availability may change; do not
claim bitwise score reproduction from this code-only release.
