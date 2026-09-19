#!/bin/bash
# Stability-extrapolation sweep on the civo cluster: one array task per arm, one H200 NVL GPU each
# (see stability_extrapolation_status.md Section 5 and runs/sweep_stability.sh).
#
#   sbatch --array=0-3 slurm/stab_sweep.slurm      # headline four: capped Poisson x {learned, 0.25, 1.0, 1.25}
#   sbatch --array=4-5 slurm/stab_sweep.slurm      # capped Poisson x {0.5, 0.75}
#   sbatch --array=6-11 slurm/stab_sweep.slurm     # all six fixed-depth arms
#
# Each task: train (resumes from a partial checkpoint if one exists, skips if TRAIN_DONE) -> depth curve ->
# eval.py cross-check at T=8,16,64. On preemption/timeout Slurm sends USR1 5 min early, train.py writes an
# emergency checkpoint, and the task is requeued; a requeued/resubmitted task simply resumes.
# QOS h200-gpupu allows 4 GPUs per user on this partition, so at most 4 arms run at once; the rest queue.
#SBATCH --job-name=stab
#SBATCH --account=mlrg_nsg
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=./slurm_logs/%x-%A_%a.out
#SBATCH --error=./slurm_logs/%x-%A_%a.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@300

PROJECT_DIR="${PROJECT_DIR:-/users/scro3801/repos/parcae-stability}"
export CONFIG="${CONFIG:-launch_configs/parcae-small-140m-stab-civo.yaml}"
export BASE_OUT="${BASE_OUT:-/datasets/scro3801/parcae-outputs/stab}"
export VAL_DIR="${VAL_DIR:-/datasets/scro3801/parcae-data/fineweb-edu-shuffled/val/data/val/train}"
GPUS=1

# task id -> (scheme, rho)
SCHEMES=(poisson-truncated-full poisson-truncated-full poisson-truncated-full poisson-truncated-full
         poisson-truncated-full poisson-truncated-full
         fixed fixed fixed fixed fixed fixed)
RHOS=(learned 0.25 1.0 1.25  0.5 0.75  learned 0.25 0.5 0.75 1.0 1.25)
i="${SLURM_ARRAY_TASK_ID:?run with --array}"
SCHEME="${SCHEMES[$i]}"; RHO="${RHOS[$i]}"

cd "${PROJECT_DIR}" || exit 1
source .venv/bin/activate
export PYTHONPATH=. OMP_NUM_THREADS=1 WANDB_MODE=offline NCCL_DEBUG=WARN
export SLURM_NTASKS_PER_NODE=${GPUS} SLURM_JOB_NUM_NODES=1   # recpre/settings.py derives `devices` from these
export TORCHRUN="torchrun --standalone --nproc_per_node=${GPUS}"
export PYTHON=python
mkdir -p "${BASE_OUT}"

echo "== job ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}) on $(hostname) $(date)"
echo "== arm: scheme=${SCHEME} rho=${RHO} -> ${BASE_OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; torch.zeros(1).cuda()" || { echo "CUDA init failed on $(hostname)"; exit 1; }

# Forward Slurm's early-warning signal to train.py (emergency checkpoint), then let the requeue resume us.
trap 'echo "USR1: asking train.py for an emergency checkpoint $(date)"; pkill -USR1 -f scripts/train.py 2>/dev/null || true; sleep 240' USR1

bash runs/sweep_stability.sh "${GPUS}" "${SCHEME}" "${RHO}" &
wait $!
EXIT=$?
echo "== arm scheme=${SCHEME} rho=${RHO} exited ${EXIT} at $(date)"
exit ${EXIT}
