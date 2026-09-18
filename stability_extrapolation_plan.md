# Injection spectral radius vs. test-time recurrence extrapolation in Parcae

**Status:** plan drafted 2026-09-18 by Hugh Blayney with Claude; code implemented and unit-tested 2026-09-18 (branch `exp/stability-extrapolation`); training smoke test partially verified (see next-steps Section 0). Sweep not yet run. Read `stability_extrapolation_next_steps.md` first; the corrections block below overrides the sections it names.
**Repo:** https://github.com/sandyresearch/parcae (this file lives at the repo root; branch off `main` at or after `69284c1`).
**Paper:** Prairie, Novack, Berg-Kirkpatrick, Fu. *Parcae: Scaling Laws For Stable Looped Language Models.* arXiv 2604.12946.

> ## Corrections found while implementing (2026-09-18)
>
> These were verified against the code and supersede the corresponding statements below.
>
> 1. **Section 4.1, `logger_name: csv`.** Upstream `setup_fabric` only accepted `wandb` and raised otherwise. A `csv` option
>    (lightning `CSVLogger`, metrics at `{out_dir}/logs/{run_name}/metrics.csv`) is now implemented and is the default in the
>    sweep config.
> 2. **Section 5, "the loaded state dict having no `A_log` will error loudly".** False: both eval loaders used
>    `load_state_dict(strict=False)`, which silently leaves a freshly initialised `A_log` (decay 0.447) in place. They now go
>    through `receval.models.parcae.load_state_dict_checked`, which raises on any missing *parameter*.
> 3. **Section 4.2, eval loop.** `scripts/eval.py` writes `{out_dir}/eval/{eval_name}.json`; with a fixed `eval_name` every
>    depth overwrote the same file. The sweep script passes `--eval_name val_loss_T{T}`. It also uses a `TRAIN_DONE` marker
>    and `--resume true` on partial checkpoints instead of "skip if any checkpoint exists".
> 4. **Section 5, held-out depth curve.** `scripts/check_state_norm.py` runs one trajectory to `max(depths)` and reads out the
>    loss at every depth along the way (identical to separate runs with `num_steps = T`, same `h0` draw), plus `||h_T||`,
>    `||h_T - h_{T-1}||` and `cos(h_T, h_{T-1})`. This replaces 14 `eval.py` runs per arm; `eval.py` is kept for a few depths
>    as a cross-check.
> 5. **Section 3, data volume.** Measured with the Parcae tokenizer: a `sample-100BT` train shard is 55.1M tokens (53,248
>    docs, 254.7 MB text, 4.62 bytes/token); a `val` shard is 10.4M tokens (10,000 docs). 2.8B tokens needs 51 shards' worth,
>    not 60 shards "to confirm". `ParquetStreamPure` partitions files across ranks with stride `max(8, world_size)` (upstream
>    `plan_for_later_rank_expansion_to=8`), so with N < 8 GPUs only shards with index mod 8 < N are ever read, and each rank
>    needs >= 1 file for train *and* val. New CLI setting `parquet_rank_expansion` (default 8; set 0 for stride = world_size)
>    lets every downloaded shard be used. Required download for 2.8B tokens on N GPUs: `N * ceil(2.8e9 / (N * 55.1e6))`
>    train shards with `parquet_rank_expansion: 0` (52 shards, 4.9 GB, for N = 4), plus >= N val shards.
> 6. **Section 4.4 / 8, A10 GPUs.** Upstream `recpre/monitor.py::_get_peak_flops` raised for "NVIDIA A10" (and any GPU not in
>    its table) at the first log step. A10/A10G/L4/L40S/A30 are added and unknown GPUs now log MFU as nan with a warning.
> 7. **Section 1.2, per-batch sampler.** In `randomized_iteration_sampler` (only used with `recurrent_iteration_method:
>    per-batch`, which no shipped Parcae config uses) `poisson-truncated-full` returns `n = Poisson draw`, `k = min(n, s)`
>    without subtracting `k`, so the loop runs `n + k` steps (mean ~12). The per-sequence path used here subtracts `k`. Left
>    as is; `max_recurrence` caps `n` in that path.
> 8. **Section 6, Step-0 probe: done** (`results/stability_probe/*.json`, 100 val docs, released `parcae-140m`). The pre-norm
>    readout does hide geometric growth: at rho = 1.25, `||h_T||` grows as 1.25^T (3e8 at T = 64) with no non-finite values,
>    `cos(h_T, h_{T-1})` locks to 1.0000 by T = 8, and PPL plateaus at ~5.1e3 from T ~ 32 rather than diverging. rho = 1.0 keeps
>    degrading (PPL 9.6e4 at T = 64), rho = 0.5 degrades steadily (526), rho = 0.25 is flat (~27), learned control saturates at
>    19.44 from T = 12. Interpretation for the trained sweep: "diverges" should be read as "plateaus at a worse level"; watch
>    `cos(h_T, h_{T-1})` and `||h_T||`, not just PPL.
> 9. **Section 4.1, `save_step_interval`.** Set to 2048 (about two mid-run checkpoints, ~2 GB each) so preempted jobs can
>    resume; 100000 would restart preempted runs from scratch.
>
> Section 7 of `stability_extrapolation_next_steps.md` lists every file the implementation touched.

This document is written so that a fresh Claude Code instance on a different cluster can implement the code changes, run the sweep, and produce the analysis without further context. Section 1 is the science; Sections 2 to 6 are the recipe; Section 7 is what to report back.

---

## 1. Question, background, and hypotheses

### 1.1 Question

Does the spectral radius rho of the linear state-carry term in Parcae's input injection determine whether a looped model keeps good perplexity when run for more recurrences than it was trained on?

### 1.2 Facts established while planning (verified against code and checkpoints, not conjecture)

- **The injection step.** `DiagonalInjection` in `parcae_lm/modules/injection.py` computes, per recurrence,
  `x_{t+1} = decay * x_t + dt * (e @ B.T)` with `dt = softplus(dt_bias)`, `decay = exp(-dt * exp(A_log))`.
  `A_log`, `dt_bias` are learned per channel of the recurrent state (768-d for the 140M model). The exp-of-exp form forces every channel of `decay` into (0, 1). The spectral radius rho = max(decay) is therefore learned, per channel, and can never reach 1. There is no config field that sets it; only the init (`ssm_decay`, default sqrt(0.2) = 0.447) is configurable.
- **What released models learn.** From the HF checkpoints `SandyResearch/parcae-{140m,370m,770m,1.3b}`:

  | model | mean decay | median | 95th pct | max channel | ‖dt·B‖₂ |
  |---|---|---|---|---|---|
  | 140M | 0.28 | 0.28 | 0.32 | 0.61 | 80 |
  | 370M | 0.20 | 0.19 | 0.26 | 0.60 | 159 |
  | 770M | 0.17 | 0.16 | 0.25 | 0.80 | 396 |
  | 1.3B | 0.16 | 0.15 | 0.25 | 0.92 | 291 |

  Training drives the carry well below the 0.447 init: the model mostly rebuilds the state each loop from the injection and the nonlinear core.
- **Training depth sampling.** The shipped 140M config uses `sampling_scheme: poisson-truncated-full` and `recurrent_iteration_method: per-sequence` (`parcae_lm/models/parcae/parcae.py`, `_sample_batch_depths`). Per sequence: `T ~ max(Poisson(mean_recurrence=8), 1)`, `k = min(T, mean_backprop_depth=4)` steps with gradient, `T - k` without. The micro-batch loops to the max T in the batch with a `torch.where` mask. This differs from Huginn, which samples the no-grad count from a Poisson-lognormal with mean `mu_rec - mu_bwd` and fixes k (paper Appendix H). Eval always runs exactly `mean_recurrence` steps.
- **T = 1..16 is inside the training region for Poisson(8).** P(T <= 16) = 0.9963; the micro-batch max T averages 12 to 14. The paper already evaluates Parcae to T = 24 and shows saturation (Section 5.3, Appendix O). "Beyond the training region" therefore needs a hard cap on training T and evaluation well past it.
- **rho of the linear part is not the contraction factor of the loop.** The per-step map is `h -> R(A h + B e)` where `R` is two pre-norm transformer blocks with residuals, so the Jacobian is `(I + J_R) A`, not `A`. Because the blocks are pre-norm, `R`'s output magnitude is roughly bounded independent of ‖h‖, and the final readout goes through `ln_f`. Consequence: for rho > 1 the state norm grows geometrically but its *direction* is dominated by the earliest contributions (the random `h0` and the first injections), and the normalized readout may plateau at a worse level rather than diverge. Treat "diverges" as one hypothesis, not the framing.
- **The paper does not do this experiment.** Figure 3 tracks rho of an *unconstrained* concatenation RDM during training and shows divergent runs learn rho >= 1 (a training-stability result). All test-time curves (Sections 5.3, Appendix L/O) are Parcae with learned rho < 1. rho is never fixed or varied, and no unstable model is evaluated past its training depth.

### 1.3 Design decisions (and the reasoning, so they can be revisited)

1. **Replace the learned per-channel decay with a hard-coded scalar rho** for the sweep arms. Keep `dt * B` as the injection so the injection scale at init (0.805·I) is identical across arms; only the state multiplier changes. `A_log` is simply not created for these arms.
2. **Arms: rho in {0.25, 0.5, 0.75, 1.0, 1.25} plus the unmodified learned control.** 0.25 is the "matched scalar" near what the model learns. 1.0 is marginally stable, not unstable. Without 1.25 there is no arm on the unstable side of the LTI condition, and the headline claim ("unstable ones diverge") cannot be tested. If 1.25 fails to train, replace with 1.1 and record the failure as a result. The learned control costs one run and is the paper's setting.
3. **Two sampling schemes.** (a) `poisson-truncated-full` with a new hard cap `max_recurrence: 16`; (b) `fixed` (exactly T = 8, 4 grad steps). The cap removes only 0.37% of Poisson(8) samples, so (a) is behaviourally the paper's recipe, but it makes "T > 16 is extrapolation" exact. (b) makes the training region a single point so any rho-dependence shows on both sides of T = 8.
4. **Random initial state stays** (`state_init: like-init`) for parity with the paper. Note it is a confound for rho >= 1: the noise in `h0` is never forgotten at rho = 1 and grows at 1.25. Optional extra arm: rho = 1.0 with `state_init: zero`.
5. **Token budget: 2.8B tokens per run first** (about 20 tokens/param, a quarter of the paper's 11.2B). Confirm interesting results at 11.2B for a subset.
6. **Truncated backprop stays at `mean_backprop_depth: 4`** in both schemes, so gradient depth is identical across arms.

### 1.4 Hypotheses and what would falsify them

| ID | Hypothesis | Prediction | Falsified if |
|---|---|---|---|
| H1 | Stable arms extrapolate flat | Under capped Poisson, rho <= 0.75 arms have PPL(T) within ~1% of PPL(16) for all T in 24..64 | PPL(64)/PPL(16) > 1.03 for rho <= 0.75 |
| H2 | Marginal arm drifts | rho = 1.0 PPL rises slowly and monotonically past 16 | flat like H1, or blows up like H3 |
| H3 | Unstable arm degrades or plateaus worse | rho = 1.25 PPL past 16 either explodes, or plateaus at a clearly worse level than at 16, and ‖h_T‖ grows as 1.25^T | rho = 1.25 matches rho = 0.5 within noise past 16 |
| H4 | Fixed-T=8 training exposes rho on both sides | All fixed arms degrade away from T = 8; degradation for T > 8 is smaller for smaller rho | no ordering by rho |
| H5 | In-distribution cost of a fixed scalar | At T = 8 fixed-rho arms are within ~3% PPL of the learned control at rho = 0.25, cost growing with rho | fixed-rho arms match or beat learned at all rho |
| H0 | Null | Extrapolation shape is independent of rho | H1 to H4 all show no rho ordering. This would mean the nonlinear core, not the linear carry, governs extrapolation, and is itself a useful result |

Record refuted hypotheses; do not delete them.

---

## 2. Code changes (target: about 30 lines, none in the training loop)

All paths are relative to the repo root. Make these on a branch, e.g. `exp/stability-extrapolation`.

### 2.1 Config fields — `parcae_lm/models/parcae/config.py`

Add to the `ParcaeConfig` dataclass, next to the other sampling fields:

```python
    # --- stability-extrapolation experiment knobs ---
    injection_fixed_decay: Optional[float] = None  # if set, A_bar = this scalar for every channel; A_log is not created
    max_recurrence: Optional[int] = None           # if set, sampled training depth T is clamped to <= this
```

### 2.2 Fixed scalar decay — `parcae_lm/modules/injection.py`

Modify `DiagonalInjection`:

```python
class DiagonalInjection(nn.Module):
    def __init__(self, config, layer_id: int = 0):
        super().__init__()
        ...  # existing asserts, state_dim, input_dim
        self.fixed_decay = getattr(config, "injection_fixed_decay", None)
        if self.fixed_decay is None:
            self.A_log = nn.Parameter(torch.empty(state_dim))
            config.init.fn("ssm_A_log", layer_id)(self.A_log)
            self.A_log._no_weight_decay = True
        else:
            self.A_log = None
        # dt_bias and B: unchanged from upstream

    def _decay(self):
        if self.fixed_decay is not None:
            return self.fixed_decay
        dt = nn.functional.softplus(self.dt_bias)
        return torch.exp(-dt * torch.exp(self.A_log))

    def forward(self, x_t, e):
        dt = nn.functional.softplus(self.dt_bias)
        return x_t * self._decay() + dt * (e @ self.B.T)

    @torch.no_grad()
    def get_spectral_norm(self) -> float:
        d = self._decay()
        return float(d) if isinstance(d, float) else d.max().item()

    @torch.no_grad()
    def get_contraction_factor(self) -> float:
        d = self._decay()
        return float(d) if isinstance(d, float) else d.mean().item()
```

Notes:
- `train.py` already builds DDP with `find_unused_parameters=True` (line ~172), so parameter-set differences between arms are harmless.
- Optimizer grouping is by parameter shape/name via `recpre.optim.get_muon_param_groups_from_config`; `dt_bias` and `B` land in AdamW as before. Check the printed AdamW/Muon param counts at startup differ from the control by exactly 768.

### 2.3 Cap on sampled depth — `parcae_lm/models/parcae/parcae.py`

Two places. `torch.clamp(..., max=None)` is a no-op, so the default config is unchanged.

In `randomized_iteration_sampler` (per-batch path), the `poisson-truncated-full` branch:

```python
            elif "poisson-truncated-full" in self.config.sampling_scheme:
                n = torch.clamp(
                    torch.poisson(torch.tensor([t + s], dtype=torch.float), generator=n_generator),
                    min=1, max=self.config.max_recurrence,
                )
                k = torch.clamp(n, max=s)
```

In `_sample_batch_depths` (per-sequence path, the one the 140M config uses):

```python
        if "poisson-truncated-full" in self.config.sampling_scheme:
            total = torch.clamp(
                torch.poisson(torch.full(shape, float(t + s)), generator=generator),
                min=1, max=self.config.max_recurrence,
            )
            k = torch.clamp(total, max=s)
            n = total - k
```

The `fixed` branch already returns `n = t = 4`, `k = s = 4`, i.e. exactly T = 8 with 4 gradient steps; nothing to change.

### 2.4 Telemetry — `parcae_lm/models/parcae/parcae.py`, `compute_extreme_metrics`

Today this function only handles linear adapters, so the learned control's rho is never logged. Add at the end, before `return metrics`:

```python
        if hasattr(adapter, "get_spectral_norm"):
            metrics["extreme/adapter_decay_max"] = adapter.get_spectral_norm()
            metrics["extreme/adapter_decay_mean"] = adapter.get_contraction_factor()
```

and set `extreme_telemetry: true` in the launch config (requires `model_telemetry: true`, already on). Also look at `monitor_module` for the recurrent-state-norm metric the paper plots (‖h_T‖); if present, make sure it is logged, since H3 predicts its growth.

### 2.5 Unit checks before any training (CPU is fine)

```python
import torch, parcae_lm
# 1. cap
cfg = parcae_lm.create_config("parcae-small-140m"); cfg.vocab_size = 32768; cfg.prelude_norm = True
cfg.max_recurrence = 16
m = cfg.construct_model(); m.train(); m.step = 0
n, k = m._sample_batch_depths(100_000); assert int((n + k).max()) <= 16 and int((n + k).min()) >= 1
assert abs(float((n + k).float().mean()) - 8.0) < 0.05
# 2. fixed decay
cfg2 = parcae_lm.create_config("parcae-small-140m"); cfg2.vocab_size = 32768; cfg2.prelude_norm = True
cfg2.injection_fixed_decay = 1.25
m2 = cfg2.construct_model()
assert m2.transformer.adapter.A_log is None and "transformer.adapter.A_log" not in m2.state_dict()
assert m2.transformer.adapter.get_spectral_norm() == 1.25
x = torch.randint(0, 32768, (2, 64)); out = m2(x, labels=x, num_steps_pair=torch.tensor([4, 4]))
out["loss"].backward(); assert m2.transformer.adapter.B.grad is not None
# 3. default unchanged
cfg3 = parcae_lm.create_config("parcae-small-140m"); cfg3.vocab_size = 32768; cfg3.prelude_norm = True
m3 = cfg3.construct_model(); assert m3.transformer.adapter.A_log is not None
```

---

## 3. Environment and data on the target cluster

- **Environment.** Either the project Docker image `ghcr.io/sandyresearch/parcae` (PyTorch 2.11, used by `launch_job.slurm`) or a venv with PyTorch >= 2.4 and `pip install -e .`. The training script also worked in a plain venv with torch 2.5.1. Run scripts with `PYTHONPATH=.` if the package is not installed. `scripts/train.py` needs a visible GPU even for `--help`.
- **`launch_job.slurm` is Docker + sudo based.** If the cluster does not allow that, write a plain sbatch script that runs `torchrun --standalone --nproc_per_node=$N scripts/train.py ...` directly (see Section 4.2).
- **Data.** Configs point to `/resource/data/fineweb-edu-100b-shuffle/{train,val}` and `/resource/tokenizer/parcae_tokenizer`; adjust to local paths. Source: HF dataset `SandyResearch/fineweb-edu-shuffled`, split dirs `data/sample-100BT/train/shard_NNNNN.parquet` (about 94 MB each, ~1000 shards, ~94 GB total) and `data/val/train/shard_NNNNNN.parquet` (218 shards, 7.3 GB). `scripts/download_data.py fineweb-100bt` grabs everything; for a 2.8B-token sweep, download a subset instead:

  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("SandyResearch/fineweb-edu-shuffled", repo_type="dataset", local_dir=DATA,
      allow_patterns=["data/sample-100BT/train/shard_000[0-5]*.parquet",   # 60 shards, ~5.6 GB
                      "data/val/train/shard_00000[0-9].parquet"])          # 10 val shards
  ```

  Tokens per shard were not measured; tokenize one shard with the Parcae tokenizer to confirm 60 shards >= 2.8B tokens (add more if not). The loader (`recpre/huggingface_dataset.py`) globs `**/*.parquet` under `data_dir`, so point `train_data.data_dir` at `.../data/sample-100BT/train` and `val_data.data_dir` at `.../data/val/train`.
- **Tokenizer.** `snapshot_download("SandyResearch/parcae-tokenizer", local_dir=TOK)`; set `tokenizer_path: TOK`.
- **Baseline reference model** for the Step-0 probe: `parcae_lm.from_pretrained("SandyResearch/parcae-140m")`.

---

## 4. Training

### 4.1 Launch config — `launch_configs/parcae-small-140m-stab.yaml`

Copy `launch_configs/parcae-small-140m.yaml` and change only these keys:

```yaml
run_name: parcae-140m-stab          # overridden per arm on the CLI
out_dir: outputs/stab               # overridden per arm on the CLI
max_tokens: 2_800_000_000
micro_batch_size: 8                 # 24 GB GPUs (A10): 8; 80 GB (H100/A100): 32 as upstream
# world_batch_size stays 256 -> gradient accumulation fills the gap
partial_depth_eval: [1, 2, 4, 12, 16, 24, 32, 64]   # 8 is appended automatically
eval_iters: 16
eval_step_interval: 512
save_step_interval: 100000          # only save at the end unless you want mid-run ckpts
save_last_step: true
extreme_telemetry: true
logger_name: csv                    # or wandb if a key is available on the node
model_overwrite:
  vocab_size: 32768
  prelude_norm: true
  max_recurrence: 16                # ignored by the fixed scheme
data_config: ...                    # local paths (Section 3)
tokenizer_path: ...
```

Everything else (MuonAdamW, lr 8e-3, trapezoid schedule with 50% cooldown, bf16-mixed, `compile_model: true`, seed 42, block size 2048) stays as upstream so the control matches the paper.

### 4.2 Sweep script — `runs/sweep_stability.sh`

```bash
#!/bin/bash
# Usage: bash runs/sweep_stability.sh <GPUS> [SCHEMES] [RHOS]
set -e
GPUS="${1:-4}"
SCHEMES="${2:-poisson-truncated-full fixed}"
RHOS="${3:-learned 0.25 0.5 0.75 1.0 1.25}"
CONFIG=launch_configs/parcae-small-140m-stab.yaml
VAL_DIR=/path/to/data/val/train          # edit
T_LIST="1 2 3 4 6 8 10 12 16 20 24 32 48 64"
export OMP_NUM_THREADS=1
for scheme in $SCHEMES; do
  for rho in $RHOS; do
    tag="${scheme%%-*}_rho${rho}"        # poisson_rho0.5, fixed_rholearned, ...
    OUT="outputs/stab/${tag}"
    EXTRA=""
    [[ "$rho" != "learned" ]] && EXTRA="--model_overwrite.injection_fixed_decay $rho"
    if ! ls "$OUT"/checkpoints-*/step-* &>/dev/null; then
      mkdir -p "$OUT"
      torchrun --standalone --nproc_per_node=$GPUS scripts/train.py \
        --config $CONFIG --out_dir "$OUT" --run_name "stab-${tag}" --resume false \
        --model_overwrite.sampling_scheme "$scheme" $EXTRA \
        --logger_group stability-extrapolation 2>&1 | tee "$OUT/train.log"
    fi
    for T in $T_LIST; do
      [[ -f "$OUT/eval_T${T}.log" ]] && continue
      torchrun --standalone --nproc_per_node=$GPUS scripts/eval.py \
        --out_dir "$OUT" --config eval_configs/eval-val-loss.yaml \
        --tasks.bpb.val_data_dir "$VAL_DIR" --override_mean_recurrence $T \
        2>&1 | tee "$OUT/eval_T${T}.log"
    done
  done
done
```

Nested `--model_overwrite.<field>` overrides on the CLI are already used by `runs/sweep_recurrence.sh`, so no parser change is needed.

### 4.3 Priority order (do not launch all 12 at once)

1. **Smoke run** (one GPU, ~5 min): `--max_tokens 0 --max_steps 30 --eval_step_interval 10 --micro_batch_size 4` for one arm with `injection_fixed_decay 1.25` and `sampling_scheme fixed`. Confirm: loss decreases, `val_ppl_{T}` keys for every T in `partial_depth_eval` appear in the log, `extreme/adapter_decay_max` logs 1.25, no NaN, tokens/s reported.
2. **Headline four** (capped Poisson): `learned`, `0.25`, `1.0`, `1.25`. This alone tests H1 to H3 and H5.
3. Remaining capped-Poisson arms: `0.5`, `0.75`.
4. Fixed-T=8 arms, in the same order (H4).
5. Optional: rho = 1.0 with `--model_overwrite.state_init zero`; rho = 1.1 if 1.25 diverged; 11.2B-token repeats of the two most informative arms.

### 4.4 Cost reference (measure on the target hardware before committing)

Measured on one NVIDIA A10 (24 GB), `torch.compile`, micro-batch 8, seq 2048, 4 no-grad + 4 grad steps:

| setting | tokens/s | peak mem |
|---|---|---|
| T = 8, compiled | 45.7k | 10.0 GiB |
| T = 12, compiled | 39.8k | 10.0 GiB |
| T = 8, eager | 19.5k | 17.6 GiB |

Per-sequence sampling runs each micro-batch to its max T (12 to 14 with the cap), so budget ~38k tokens/s per A10 solo, ~34k per GPU under DDP. Approximate FLOPs: 1.5e9 per token at T = 8 (including the tied LM head), so 2.8B tokens ≈ 4.3e18 FLOPs and 11.2B ≈ 1.7e19 FLOPs per run. Wall-clock for 2.8B tokens: ~6 h on 4 A10s; ~23 h for 11.2B. H100-class GPUs were not benchmarked; a 4 to 6x per-GPU speedup over an A10 is a reasonable prior, so measure it with the smoke run. Twelve 2.8B-token runs ≈ 5e19 FLOPs total.

---

## 5. Evaluation

Two sources, both needed:

1. **Training-time curve (free).** `partial_depth_eval` makes `validate()` in `scripts/train.py` log `val_loss_{T}` / `val_ppl_{T}` for every listed T at each eval interval and at the end. This gives the final curve *and* its evolution over training (does the T = 64 loss of the rho = 1.25 arm get worse as training proceeds?). Parse from the CSV/W&B logs.
2. **Held-out eval at the end (bigger sample).** `scripts/eval.py --override_mean_recurrence T` with `eval_configs/eval-val-loss.yaml` (1000 documents, up to 5 files). It routes through `receval/models/parcae.py::forward_for_generation`, which calls the same `core_block_forward` and therefore the modified adapter. Verify once that the run's config (with `injection_fixed_decay`) is what `receval` reconstructs; the loaded state dict having no `A_log` will error loudly if not.

Also collect **state norm vs T** with a small script (`scripts/check_state_norm.py`): load each final checkpoint, run 32 validation sequences at each T in the list with `num_steps_pair=torch.tensor([T, 0])`, and record mean ‖h_T‖ (the tensor returned first from `iterate_forward`) and ‖h_T − h_{T−1}‖. H3 predicts ~1.25^T growth; H1 predicts saturation.

Sanity checks on the numbers:
- Control arm at T = 8 should land near the paper's 140M / 11.2B-token quality scaled for the smaller budget; the shipped `parcae-140m` checkpoint's decay stats (mean 0.28, max 0.61) give a reference for what the learned control should converge towards.
- For the capped-Poisson control, the T = 1..16 curve should reproduce the paper's saturating-exponential shape (Figure 8).
- `expected_total_recurrence` and the logged per-step depth should never exceed 16 in capped-Poisson runs.

---

## 6. Optional Step 0: a cheap inference-only probe (minutes, one GPU)

Before training, load `SandyResearch/parcae-140m`, monkeypatch the adapter's forward to use a scalar rho in {0.25, 0.5, 1.0, 1.25} in place of the learned decay (keep `dt * B`), and record PPL and ‖h_T‖ for T in 1..64 on ~100 validation documents. The weights were trained at rho ≈ 0.28, so this does not test the training hypothesis, but it shows directly whether the pre-norm readout hides geometric state growth (Section 1.2, last bullet), which decides how the trained sweep should be read.

---

## 7. Analysis and reporting

Plots (one figure each, both sampling schemes as side-by-side panels, one line per rho, log2 x-axis, vertical line at the training cap T = 16 or the fixed T = 8):

1. Val PPL vs T.
2. PPL normalized by each arm's own PPL at T = 8 (separates in-distribution quality from extrapolation shape).
3. ‖h_T‖ vs T (log y).
4. `val_ppl_64` vs training step for each arm (drift during training).
5. For the learned control: `extreme/adapter_decay_max` and `_mean` vs training step.

Table: PPL at T in {1, 4, 8, 16, 32, 64} per arm, plus PPL(64)/PPL(16) and PPL(8)/PPL_control(8).

Write results as lab-book entries against the hypotheses in Section 1.4 with the falsification criteria applied literally. Report divergent or NaN runs as results with the step at which they failed. Keep the per-arm logs, final checkpoints, and the eval JSONs; everything in `outputs/stab/`.

---

## 8. Known pitfalls

- `scripts/train.py` needs `PYTHONPATH=.` (or `pip install -e .`) and a visible GPU even to parse args.
- `max_steps` is only honoured if `max_tokens: 0`.
- On 24 GB GPUs use `micro_batch_size: 8`; 16 may fit compiled (10 GiB at 8) but was not tested.
- `compile_model: true` costs ~40 s at startup and roughly doubles throughput; keep it.
- `logger_name: wandb` needs `WANDB_API_KEY` on the compute node; `csv` needs nothing.
- The default `recurrence_schedule: linear` with `cirriculum_steps: 0` is effectively constant; leave it.
- The eval config's `val_data_dir` is `null` upstream and must be passed.
- Do not merge `partial_depth_eval` changes upstream (the settings file says so); this is an experiment branch.
