#!/bin/bash
# Stability sweep smoke test on the civo cluster (stability_extrapolation_next_steps.md, Sections 0 + 3).
# One H200 NVL GPU, ~30-45 min: unit tests -> 30 training steps on the real data (checkpoint at the last step)
# -> scripts/eval.py reload with --override_mean_recurrence 16 -> scripts/check_state_norm.py depth curve
# -> metrics.csv sanity summary. Outputs go to /datasets/scro3801/parcae-outputs/stab/smoke (delete afterwards).
#
#   sbatch slurm/stab_smoke.slurm              # then: tail -f slurm_logs/stab-smoke-<jobid>.out
#
# Partition h200 = eight separate 1-GPU nodes (14 CPU, 160 GB); QOS h200-gpupu caps each user at 4 GPUs there,
# independently of the a100/b200 caps that the rec1 control runs are using.
#SBATCH --job-name=stab-smoke
#SBATCH --account=mlrg_nsg
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=./slurm_logs/%x-%j.out
#SBATCH --error=./slurm_logs/%x-%j.out

PROJECT_DIR="${PROJECT_DIR:-/users/scro3801/repos/parcae-stability}"
CONFIG="${CONFIG:-launch_configs/parcae-small-140m-stab-civo.yaml}"
OUT="${OUT:-/datasets/scro3801/parcae-outputs/stab/smoke}"
VAL_DIR="${VAL_DIR:-/datasets/scro3801/parcae-data/fineweb-edu-shuffled/val/data/val/train}"
MBS="${MBS:-32}"          # micro_batch_size to calibrate; world_batch_size stays 256
STEPS="${STEPS:-30}"

cd "${PROJECT_DIR}" || exit 1
source .venv/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 WANDB_MODE=offline NCCL_DEBUG=WARN
export SLURM_NTASKS_PER_NODE=1 SLURM_JOB_NUM_NODES=1   # recpre/settings.py derives `devices` from these
echo "== job ${SLURM_JOB_ID} on $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "python: $(which python)"; python -c "import torch; print('torch', torch.__version__, 'cuda ok:', torch.zeros(1).cuda().item() == 0)" || exit 1

set -x
rm -rf "${OUT}"; mkdir -p "${OUT}"

echo "== 0. unit tests (CPU)"
python tests/test_stability_knobs.py || { echo "UNIT TESTS FAILED"; exit 1; }

echo "== 1. train ${STEPS} steps, fixed rho=1.25, micro_batch_size=${MBS}"
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --config "${CONFIG}" --out_dir "${OUT}" --run_name stab-smoke --resume false \
  --max_tokens 0 --max_steps "${STEPS}" --eval_step_interval 10 --eval_iters 2 \
  --micro_batch_size "${MBS}" --save_step_interval 100000 \
  --model_overwrite.sampling_scheme fixed --model_overwrite.injection_fixed_decay 1.25 \
  2>&1 | tee "${OUT}/train.log"
grep -q "Exiting training orderly after completion" "${OUT}/train.log" || { echo "TRAINING DID NOT FINISH"; exit 1; }

echo "== 1b. checkpoint exists?"
ls -la "${OUT}"/checkpoints-*/step-* || { echo "NO CHECKPOINT SAVED"; exit 1; }

echo "== 2. eval.py reload + --override_mean_recurrence 16 (expect 'Overriding mean_recurrence: 8 -> 16', no load_state_dict_checked error)"
torchrun --standalone --nproc_per_node=1 scripts/eval.py --out_dir "${OUT}" --config eval_configs/eval-val-loss.yaml \
  --eval_name val_loss_T16 --tasks.bpb.val_data_dir "${VAL_DIR}" --tasks.bpb.max_samples 50 \
  --override_mean_recurrence 16 2>&1 | tee "${OUT}/eval_T16.log"
cat "${OUT}/eval/val_loss_T16.json"

echo "== 3. depth curve on the trained run (loss at depth 16 should be within ~0.01 nats of eval/val_loss_T16.json)"
python scripts/check_state_norm.py --out_dir "${OUT}" --val_data_dir "${VAL_DIR}" \
  --max_samples 50 --depths 1,2,4,8,16,32,64 2>&1 | tee "${OUT}/depth_curve.log"

echo "== 4. metrics.csv summary"
python - "${OUT}/logs/stab-smoke/metrics.csv" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
print("columns:", len(df.columns), "rows:", len(df))
val_cols = sorted(c for c in df.columns if c.startswith("val_ppl_"))
print("val_ppl columns:", val_cols)
# NB: the CSV logger's names, not the ones in stability_extrapolation_next_steps.md Section 3:
# "loss" is global_loss/local_loss and "tokens_per_second" is tokens/second.
for c in ["global_loss", "extreme/adapter_decay_max", "extreme/adapter_decay_mean", "recurrent_state_norm",
          "tokens/second", "seconds/step", "max_mem_per_gpu", "max_mem_reserved_per_gpu",
          "val_time", "num_steps_no_grad", "num_steps_with_grad"]:
    if c in df.columns:
        s = df[c].dropna()
        print(f"{c:32s} n={len(s):3d} first={s.iloc[0]:.4g} last={s.iloc[-1]:.4g} min={s.min():.4g} max={s.max():.4g}")
    else:
        print(f"{c:32s} MISSING")
print(df[val_cols].dropna(how="all").to_string())
PY
set +x
du -sh "${OUT}"; echo "== done $(date). Delete ${OUT} when finished with it."
