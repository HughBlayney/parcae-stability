# Stability-extrapolation sweep: current state

**Status: 2026-09-19 02:05.** Branch `exp/stability-extrapolation`, at `8ff6bd9`. The implementation is done and
committed, the environment is built on the civo cluster, the end-to-end pipeline is verified, and **all 12 sweep
arms are queued as slurm array job `51027_[0-11]`**. Nothing remains before results except waiting for the queue.

Read this file, then `stability_extrapolation_plan.md` (its "Corrections" block first). This file replaces
`stability_extrapolation_next_steps.md`, which described the pre-cluster state and is deleted.

## 1. What is verified

The three things the old next-steps doc flagged as *never verified anywhere* are now verified, by
`slurm/stab_smoke.slurm` (job 51025, one B200, 17 min, exit 0):

| Check | Result |
|---|---|
| `tests/test_stability_knobs.py` | 7/7 pass |
| 30 training steps, fixed rho = 1.25 | loss 10.60 -> 6.62 |
| **Checkpoint written at the last step** | `step-00000232` (29 optimizer steps x 8 accumulation; the counter is micro-steps) |
| **`eval.py` reload of a fixed-decay checkpoint** | "Overriding mean_recurrence: 8 -> 16", no `load_state_dict_checked` error |
| **`check_state_norm.py` on a trained run** | depth curve written for all requested depths |
| **Cross-check** | `eval.py` T=16 loss 6.543212 vs depth curve T=16 6.5433 — **0.0001 nats**, tolerance was 0.01 |
| `val_loss_{T}` / `val_ppl_{T}` | all 9 depths present in `metrics.csv` |
| `extreme/adapter_decay_max` | exactly 1.25 on every step |
| `recurrent_state_norm` | finite, 438 -> 2753 |
| `num_steps_no_grad + num_steps_with_grad` | 4 + 4 = 8 |

The smoke run's depth curve is **dead flat** (loss 6.5431 at every T from 1 to 64) while `||h_T||` grows from
113 to 7.2e8 and `cos(h_T, h_T-1)` is 1.0000 throughout. This is a 30-step model that has not learned to use
the recurrent state, not a plumbing fault: the Step-0 probe on the **released** weights
(`results/stability_probe/*.json`) shows strongly depth-varying loss with the same code path —

| rho | T=1 | T=2 | T=4 | T=8 | T=16 |
|---|---|---|---|---|---|
| learned | 3.298 | 3.098 | 2.993 | 2.969 | 2.967 |
| 0.25 | 3.298 | 3.263 | 3.240 | 3.258 | 3.287 |
| 0.5 | 3.306 | 3.386 | 3.754 | 4.628 | 5.416 |
| 1.0 | 3.340 | 3.641 | 4.966 | 7.431 | 9.487 |
| 1.25 | 3.367 | 3.770 | 5.390 | 7.431 | 8.373 |

Apply the same check to the first real arm: if a trained low-rho arm still gives a flat curve, *that* is a bug.

## 2. Environment (built, do not rebuild)

`/datasets/scro3801/venvs/parcae-stability`, symlinked as `.venv`. Python 3.12, pinned to match the known-good
`/users/scro3801/repos/external/parcae/env_parcae`: **torch 2.13.0+cu130, lightning 2.3.0.dev20240328,
jsonargparse 4.49.0, numpy 1.26.4**. Works on B200 (sm_100); no `flash_attn` (SDPA fallback). Run from the repo
root with `PYTHONPATH=.`.

If it ever needs rebuilding: `uv pip install --python .venv/bin/python -e . "setuptools<81" -c <freeze of env_parcae>`
with `UV_CACHE_DIR=/datasets/scro3801/uv-cache`. **Do not set `UV_LINK_MODE=copy`** — the cache and the venv are
on the same NFS mount so uv can hardlink; copy mode wrote 49,176 files / 5.2 GB byte-for-byte (~20 min) and
corrupted packages twice when interrupted mid-copy.

On the **login node**, always set `OMP_NUM_THREADS`. It has 220 cores but sits at load ~350 from other users, and
torch otherwise spawns ~290 threads: a 28 s CPU test burned 29 CPU-minutes.

## 3. Cluster and data

Account `mlrg_nsg`. QOS caps are **per partition**, so jobs on different partitions do not compete:

| partition | nodes | GPUs | limit | cap |
|---|---|---|---|---|
| `h200` | civo-h200-[2-9] | 1x H200 NVL each | 2 days | 4 GPU |
| `a100` | civo-a100-1 | 8x A100-80GB | 2 days | 4 GPU |
| `b200` | civo-b200-1 | 8x B200 (183 GB) | 2 days | 4 GPU |
| `allgpus` (default) | all of the above | mixed | 12 h | 4 GPU |
| `vgg` | civo-h200-1 | 8x H200 SXM | 7 days | **no access** |

- The sweep runs on `h200`: 12 independent 1-GPU arms match 8 single-GPU nodes, it avoids competing with the
  `rec1` control runs queued on `a100`/`b200`, and it keeps **all arms on identical hardware** — arms are the
  unit of comparison, so splitting them across GPU types would put a hardware confound on the measured axis.
- `allgpus` is a second 4-GPU cap on the *same physical nodes*: `--partition=allgpus
  --exclude=civo-a100-1,civo-b200-1` gives 8 concurrent arms on matching hardware if the sweep needs compressing.
- `vgg` refuses this account ("Invalid account or account/partition combination") despite `AllowAccounts=ALL`.
  It is 8x H200 SXM with the emptiest queue on the cluster and is worth asking the admins for.
- Use `sbatch --test-only ... --wrap=hostname` for the scheduler's own start estimate before committing a job.
  Raw queue depth misleads: most pending `h200` jobs are one user blocked by their own `QOSMaxGRESPerUser` cap.

Data is already present, no downloads needed:

- train: `/datasets/scro3801/parcae-data/fineweb-edu-shuffled/sample-100BT/data/sample-100BT/train` (1822 shards)
- val: `/datasets/scro3801/parcae-data/fineweb-edu-shuffled/val/data/val/train` (218 shards)
- tokenizer: `/datasets/scro3801/parcae-data/tokenizer/parcae_tokenizer` (vocab 32768)

Home is a 50 GB volume; everything lives under `/datasets/scro3801/`. Budget ~72 GB for 12 arms.

## 4. Calibration (measured, B200, micro_batch_size 32)

- **549k tokens/s** steady state. Scaling by peak bf16 (B200 ~2250 vs H200 989 TFLOP/s) puts the H200 NVL at
  **250-350k tok/s**; confirm from the first arm rather than trusting this.
- **34.8 GB allocated / 62.5 GB reserved.** Keep `micro_batch_size` at 32: doubling it would put reserved memory
  near 125 GB against the NVL's 141 GB. `world_batch_size` stays 256 via gradient accumulation.
- **118 s per validation round** (9 depths x `eval_iters` 16), ~10 rounds per arm at `eval_step_interval` 512.
- Per arm: 2.8B tokens / 5,340 steps, so **~2.2-3.1 h train + ~20 min in-training validation + ~15 min post-train
  evals = ~3-4 h**. Twelve arms at 4 concurrent = 3 waves, **~9-12 h of compute**; wall-clock is queue-dominated.

## 5. The running sweep

`sbatch --array=0-11 slurm/stab_sweep.slurm` -> **array job `51027_[0-11]`**, submitted 2026-09-19 01:59, one
H200 NVL and 24 h per task. Slurm runs 4 at a time and queues the rest behind the QOS cap.

| tasks | scheme | rho | output under `/datasets/scro3801/parcae-outputs/stab/` |
|---|---|---|---|
| 0-3 | `poisson-truncated-full` | learned, 0.25, 1.0, 1.25 | `poisson_rho{learned,0.25,1.0,1.25}` |
| 4-5 | `poisson-truncated-full` | 0.5, 0.75 | `poisson_rho{0.5,0.75}` |
| 6-11 | `fixed` | learned, 0.25, 0.5, 0.75, 1.0, 1.25 | `fixed_rho{...}` |

Tasks 0-3 are the plan's headline four and land in the first wave. Each task runs train -> 14-depth curve ->
`eval.py` at T = 8, 16, 64, skipping any stage whose output already exists. On preemption it catches `USR1`,
writes an emergency checkpoint and requeues; resubmitting the same task resumes.

Per arm you get `train.log`, `logs/<run>/metrics.csv`, `eval/depth_curve.json` (1000 held-out docs; loss, PPL,
`||h_T||`, step norm, cosine at 14 depths), `eval/val_loss_T{8,16,64}.json`, and the final checkpoint.

**While arms are queued, do not edit the repo.** Each task reads the working tree when it launches, so queued
and requeued arms pick up any change to `runs/sweep_stability.sh`, the config or `parcae_lm/`.

`runs/sweep_stability.sh` env knobs: `CONFIG, BASE_OUT, VAL_DIR (required), TOK, DEPTHS, EVAL_T_LIST, EVAL_DOCS,
EXTRA_TRAIN, EXTRA_ARM, TORCHRUN, PYTHON`.

## 6. What remains

1. **Watch the first arm.** Record real H200 tok/s from `tokens/second`; confirm a trained low-rho arm gives a
   depth-varying curve (Section 1).
2. **Write `scripts/plot_stability.py`** (still unwritten — there were no results to test it against). Inputs are
   the per-arm files above; outputs the five plots and the table in plan Section 7.
3. **Apply the falsification criteria in plan Section 1.4 literally** and record refuted hypotheses.
4. Optional follow-ups: `EXTRA_ARM="--model_overwrite.state_init zero"` with `BASE_OUT=outputs/stab_zero` for
   rho 1.0; rho 1.1 if 1.25 fails to train; 11.2B-token repeats via `EXTRA_TRAIN="--max_tokens 11200000000"` and
   a different `BASE_OUT` (4x the train shards).
5. Delete `/datasets/scro3801/parcae-outputs/stab/smoke` (1.6 GB) when no longer wanted.

### Sanity checks before drawing conclusions

- `num_steps_no_grad + num_steps_with_grad <= 16` in every capped-Poisson `metrics.csv` row.
- `eval/val_loss_T8.json` loss within ~0.01 nats of `depth_curve.json` at depth 8 (same docs, different `h0` draw).
  The smoke run achieved 0.0001.
- `extreme/adapter_decay_max` constant at rho for fixed arms; for the learned control it should fall from 0.447
  toward ~0.3 (released model: max 0.61, mean 0.28).
- Capped-Poisson control at T = 8 vs the released `parcae-140m` (PPL 19.47 on 100 docs of `val/shard_000000`,
  `results/stability_probe/probe_rholearned.json`); expect worse at 2.8B tokens but the same shape.

### Column names in `metrics.csv` — the old doc had these wrong

The CSV logger writes **`global_loss`** / `local_loss` (not `loss`) and **`tokens/second`** (not
`tokens_per_second`). Also useful: `seconds/step`, `val_time`, `max_mem_reserved_per_gpu`.

**`model_flop_utilization` and the `FLOP/S` columns are unreliable here** and should not appear in the analysis:
the FLOP counter cannot trace this architecture ("Cannot trace model with meta tensors for flop calculation") and
falls back to an estimate, reporting an impossible MFU of 1.37. `tokens/second` is wall-clock measured and sound.

## 7. What the implementation touched (for review)

- `parcae_lm/models/parcae/config.py`: `injection_fixed_decay`, `max_recurrence` (+ validation).
- `parcae_lm/modules/injection.py`: `DiagonalInjection` fixed-scalar mode; `A_log` absent from module and state dict.
- `parcae_lm/models/parcae/parcae.py`: `max_recurrence` clamp in both samplers; `extreme/adapter_decay_{max,mean}`.
- `scripts/train.py`, `recpre/utils.py`: `logger_name: csv`; `parquet_rank_expansion` pass-through.
- `recpre/settings.py`: `parquet_rank_expansion` setting.
- `recpre/monitor.py`: A10/A10G/L4/L40S/A30 **and H200** peak FLOPs; unknown GPUs no longer abort training.
- `receval/models/parcae.py`, `scripts/eval.py`: `load_state_dict_checked` (missing parameters raise).
- `scripts/check_state_norm.py`: single-trajectory depth curve + state norms; Step-0 probe (`--hf_repo`,
  `--override_fixed_decay`).
- `runs/sweep_stability.sh`: per-arm train -> depth curve -> eval; only marks `TRAIN_DONE` on the orderly
  end-of-run log line, since `train.py` also exits 0 after an emergency checkpoint (fixed in `fb0013b` — before
  that, a preempted arm was recorded as finished and evaluated against a partial model).
- `launch_configs/parcae-small-140m-stab.yaml` and `-civo.yaml`, `slurm/stab_{smoke,sweep}.slurm`,
  `tests/test_stability_knobs.py`, `results/stability_probe/*.json`, `stability_extrapolation_plan.md`.

Not changed: the training loop, optimizer grouping (`dt_bias`, `B` stay in AdamW; fixed arms simply have 768
fewer parameters), the eval config, anything used by the default config (all new knobs default to upstream).
