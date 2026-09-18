# Stability-extrapolation sweep: what remains on the target cluster

**Status:** 2026-09-18. Code for `stability_extrapolation_plan.md` is implemented on branch `exp/stability-extrapolation`
and unit-tested on CPU (`tests/test_stability_knobs.py`). A 20-step training smoke run (fixed scheme, rho = 1.25, one A10,
`compile_model: true`, DDP strategy, CSV logger) was started on the dev box but **stopped at 30/40 micro-steps** because long
jobs there must go through slurm. Before it was stopped it verified: startup with `--model_overwrite.injection_fixed_decay
1.25 --model_overwrite.sampling_scheme fixed`, `AdamW`/`Muon` grouping without `A_log`, compile + first fwd/bwd, loss falling
(10.6 -> 8.3), three validation passes writing `val_loss_{T}` / `val_ppl_{T}` for T in {1,2,4,8,12,16,24,32,64} to
`metrics.csv`, `extreme/adapter_decay_max == 1.25` every step, `recurrent_state_norm` growing (438 -> 2552) and finite.
**Not yet verified anywhere (do this first, Section 0):** checkpoint save at the last step, `scripts/eval.py` reloading a
fixed-decay checkpoint (config reconstruction from `run_config.json` + the strict loader) with `--override_mean_recurrence`,
and `scripts/check_state_norm.py --out_dir` on a trained run (only its `--hf_repo` path has run). No sweep has been run.

Read this file, then the plan's "Corrections" block, then the plan itself.

## 0. First slurm job: finish the smoke test (about 15 min on one GPU)

Submit as one sbatch job (adapt the env activation and paths; the dev box smoke data is at
`/data-gauss/hugbla/parcae-stability-smoke/{train,val,tokenizer}` and `launch_configs/parcae-small-140m-stab.yaml` needs its
three `EDIT ME` paths pointed there, e.g. via a copy of the config):

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1
OUT=outputs/stab/smoke
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --config launch_configs/parcae-small-140m-stab.yaml --out_dir $OUT --run_name stab-smoke --resume false \
  --max_tokens 0 --max_steps 20 --eval_step_interval 10 --eval_iters 2 --micro_batch_size 4 --world_batch_size 8 \
  --save_step_interval 100000 --model_overwrite.sampling_scheme fixed --model_overwrite.injection_fixed_decay 1.25
# 1. checkpoint exists:            ls $OUT/checkpoints-*/step-*
# 2. eval.py reload + override:     (must print "Overriding mean_recurrence: 8 -> 16", no load_state_dict_checked error)
torchrun --standalone --nproc_per_node=1 scripts/eval.py --out_dir $OUT --config eval_configs/eval-val-loss.yaml \
  --eval_name val_loss_T16 --tasks.bpb.val_data_dir /data-gauss/hugbla/parcae-stability-smoke/val \
  --tasks.bpb.max_samples 50 --override_mean_recurrence 16
# 3. depth curve on the trained run: (loss at depth 16 within ~0.01 nats of eval/val_loss_T16.json)
python scripts/check_state_norm.py --out_dir $OUT --val_data_dir /data-gauss/hugbla/parcae-stability-smoke/val \
  --max_samples 50 --depths 1,2,4,8,16,32,64
```

With 1 train + 1 val shard this only works on **one** rank (the loader partitions files across ranks, Section 2). Delete
`outputs/stab/smoke` afterwards (checkpoint ~2 GB). Note the loss/PPL at 20 steps is meaningless; only the plumbing matters.

## 1. Environment (one-off)

- Either the project Docker image (`launch_job.slurm`) or a Python env with the `pyproject.toml` dependencies and
  `torch >= 2.4`. Verified working env on the dev box: torch 2.9 + lightning 2.3.0.dev + torchdata + jsonargparse 4.46,
  no `flash_attn` (SDPA fallback). Run everything from the repo root with `PYTHONPATH=.`.
- Sanity check without data or GPU:

  ```bash
  PYTHONPATH=. python tests/test_stability_knobs.py
  ```

## 2. Data (one-off; sizes measured, see plan corrections item 5)

Put data under your own directory. For **N GPUs** and the **2.8B-token** budget with `parquet_rank_expansion: 0`:

- train shards needed: `N * ceil(2.8e9 / (N * 55.1e6))` (N = 4: 52 shards ~4.9 GB; N = 8: 56 shards ~5.3 GB). Each rank
  reads `files[rank::N]`, so every rank must own enough files for its share of tokens; add 1 to 2 shards per rank of margin.
- val shards needed: `>= N` (each 28 MB / 10.4M tokens). Training-time validation restarts the val stream for every depth,
  so `eval_iters * micro_batch_size * 2049` tokens per rank per depth is all that is read.
- For 11.2B-token repeats: 4x the train shards.

```python
from huggingface_hub import snapshot_download
DATA = "/your/dir/fineweb-edu-shuffled"
snapshot_download("SandyResearch/fineweb-edu-shuffled", repo_type="dataset", local_dir=DATA,
    allow_patterns=[f"data/sample-100BT/train/shard_{i:05d}.parquet" for i in range(56)]
                 + [f"data/val/train/shard_{i:06d}.parquet" for i in range(8)])
snapshot_download("SandyResearch/parcae-tokenizer", local_dir="/your/dir/parcae_tokenizer")
```

Then edit the three `EDIT ME` paths at the bottom of `launch_configs/parcae-small-140m-stab.yaml`
(`train_data.data_dir` -> `.../data/sample-100BT/train`, `val_data.data_dir` -> `.../data/val/train`, `tokenizer_path`)
and add `parquet_rank_expansion: 0` to that file (or pass `--parquet_rank_expansion 0`).

If you use the Docker launcher, its `/resource/...` mounts already match the config's default paths.

## 3. Throughput / memory calibration (30 min)

Not done (the dev box measured 45k tok/s at micro-batch 4 on one A10 during the stopped smoke run, matching plan
Section 4.4). One smoke arm, one GPU, on the real data:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --config launch_configs/parcae-small-140m-stab.yaml --out_dir outputs/stab/smoke --run_name stab-smoke \
  --max_tokens 0 --max_steps 30 --eval_step_interval 10 --eval_iters 2 \
  --model_overwrite.sampling_scheme fixed --model_overwrite.injection_fixed_decay 1.25
```

Check in `outputs/stab/smoke/logs/stab-smoke/metrics.csv`: `loss` decreasing, `val_ppl_{1,2,4,8,12,16,24,32,64}` columns
present, `extreme/adapter_decay_max == 1.25`, `recurrent_state_norm` finite, `tokens_per_second`, `max_mem_per_gpu`.
Then raise `micro_batch_size` (8 on 24 GB, 32 on 80 GB) as memory allows; `world_batch_size` stays 256. Delete
`outputs/stab/smoke` afterwards (its checkpoint is ~2 GB). Budget from the dev-box A10 numbers in plan Section 4.4 if the
target GPUs are also A10s: about 6 h per 2.8B-token arm on 4 GPUs.

## 4. Job scripts (to write on the cluster)

`runs/sweep_stability.sh` does train -> depth curve -> eval.py per arm, resumes from partial checkpoints and skips finished
stages, so a plain sbatch wrapper is enough. Suggested shape (one array task per arm keeps queue time low):

```bash
#SBATCH --array=0-11    # 2 schemes x 6 rhos; map $SLURM_ARRAY_TASK_ID -> (scheme, rho)
export VAL_DIR=/your/dir/fineweb-edu-shuffled/data/val/train
export PYTHON=python TORCHRUN="torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE"
bash runs/sweep_stability.sh $SLURM_GPUS_ON_NODE "$SCHEME" "$RHO"
```

Env knobs the script understands: `CONFIG, BASE_OUT, VAL_DIR (required), TOK, DEPTHS, EVAL_T_LIST, EVAL_DOCS, EXTRA_TRAIN,
EXTRA_ARM, TORCHRUN, PYTHON`. On preemption, resubmit the same task: it passes `--resume true` when checkpoints exist.
Set `#SBATCH --signal=USR1@120` if you want the emergency-checkpoint handler in `train.py` to fire.

## 5. Run order (plan Section 4.3, unchanged)

1. Headline four, capped Poisson: `bash runs/sweep_stability.sh N "poisson-truncated-full" "learned 0.25 1.0 1.25"`.
2. `0.5 0.75` capped Poisson.
3. All six `fixed`.
4. Optional: `EXTRA_ARM="--model_overwrite.state_init zero"` with `BASE_OUT=outputs/stab_zero` for rho 1.0; rho 1.1 if 1.25
   fails to train; 11.2B-token repeats via `EXTRA_TRAIN="--max_tokens 11200000000"` and a different `BASE_OUT`.

Per arm you get: `train.log`, `logs/<run>/metrics.csv` (per-step training metrics + `val_ppl_{T}` every 512 steps),
`eval/depth_curve.json` (1000 held-out docs, loss/PPL/`||h_T||`/step norm/cosine at 14 depths), `eval/val_loss_T{8,16,64}.json`
(independent cross-check via `eval.py`), and the final checkpoint.

## 6. Analysis (plan Section 7)

Write `scripts/plot_stability.py` (not written yet: no results to test against). Inputs are the files above; outputs the five
plots and the table in plan Section 7. Sanity checks to apply first:

- capped-Poisson control at T = 8 vs the released `parcae-140m` (PPL 19.47 on 100 docs of `val/shard_000000` at T = 8 with
  the released weights, `results/stability_probe/probe_rholearned.json`); expect worse at 2.8B tokens but same shape.
- `num_steps_no_grad + num_steps_with_grad <= 16` in every capped-Poisson `metrics.csv` row.
- `eval/val_loss_T8.json` loss within ~0.01 nats of `depth_curve.json` loss at depth 8 (same docs, different `h0` draw).
- `extreme/adapter_decay_max` constant at rho for fixed arms; for the learned control it should fall from 0.447 towards ~0.3
  (released model: max 0.61, mean 0.28).

Then apply the falsification criteria in plan Section 1.4 literally and record refuted hypotheses.

## 7. What the implementation touched (for review)

- `parcae_lm/models/parcae/config.py`: `injection_fixed_decay`, `max_recurrence` (+ validation).
- `parcae_lm/modules/injection.py`: `DiagonalInjection` fixed-scalar mode; `A_log` absent from the module and state dict.
- `parcae_lm/models/parcae/parcae.py`: `max_recurrence` clamp in both samplers; `extreme/adapter_decay_{max,mean}` telemetry.
- `scripts/train.py`, `recpre/utils.py`: `logger_name: csv`; `parquet_rank_expansion` pass-through.
- `recpre/settings.py`: `parquet_rank_expansion` setting.
- `recpre/monitor.py`: A10/A10G/L4/L40S/A30 peak FLOPs; unknown GPUs no longer abort training.
- `receval/models/parcae.py`, `scripts/eval.py`: `load_state_dict_checked` (missing parameters raise).
- `scripts/check_state_norm.py`: single-trajectory depth curve + state norms; also the Step-0 probe (`--hf_repo`,
  `--override_fixed_decay`).
- `launch_configs/parcae-small-140m-stab.yaml`, `runs/sweep_stability.sh`, `tests/test_stability_knobs.py`,
  `results/stability_probe/*.json`, `stability_extrapolation_plan.md` (with corrections block).

Not changed: the training loop, optimizer grouping (`dt_bias`, `B` stay in AdamW; the fixed arms simply have 768 fewer
parameters), the eval config, anything used by the default config (all new knobs default to upstream behaviour).
