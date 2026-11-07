#!/bin/bash
# =============================================================================
# Single-node, 4-GPU training for the recurrence-ablated control runs.
# Native (no Docker): this cluster has no container runtime, so we run the repo's
# env_parcae venv directly under torchrun. Validated end-to-end on an A100 node.
#
#   CONFIG=launch_configs/parcae-small-140m-rec1.yaml sbatch launch_control.slurm
#
# Defaults to the a100 partition. Only TWO nodes on this cluster can satisfy this job spec:
#   civo-a100-1  8x A100-80GB, 96 CPU,  860 GB   (partition a100)
#   civo-b200-1  8x B200,     220 CPU, 2576 GB   (partition b200)
# NOT h200: that partition is eight SEPARATE 1-GPU nodes (14 CPU, 160 GB each), so
# --nodes=1 --gres=gpu:4 with 32 CPUs and 200 GB can never be scheduled there -- the job
# would sit PENDING forever rather than fail. (Checked 2026-09-18; civo-h200-1, the only
# 8-GPU H200 box, is in the `vgg` partition and has been DRAIN since 2026-09-15.)
#
# 4 GPUs is the hard ceiling: the a100-gpupu / h200-gpupu / b200-gpupu QOS each set
# MaxTRESPerUser=gres/gpu=4. Walltime max is 2 days on all three.
#
# Approx runtime, control runs only (recurrence=1), 4 GPUs, assuming ~35-40% MFU:
#              4xA100     4xB200
#   small       ~6 h       ~1 h
#   medium     ~43 h       ~7 h
#   large      ~7.5 d     ~30 h
#   xlarge     ~21 d      ~3.6 d    <- also needs the 350BT dataset, not 100BT
# Anything over the 2-day limit relies on --requeue plus `resume: true` in its config.
# =============================================================================
#SBATCH --job-name=parcae-rec1
#SBATCH --account=mlrg_nsg
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200000
#SBATCH --time=2-00:00:00
# Paths are relative to the directory you run sbatch from (run_controls.sh cds to the repo
# first, and also passes absolute paths that override these). %x = job name, %j = job id.
# Slurm does NOT create these directories -- they must exist at submit time.
#SBATCH --output=./slurm_out/%x-%j.out
#SBATCH --error=./slurm_err/%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# =============================================================================
# User configuration
# =============================================================================
PROJECT_DIR="${PROJECT_DIR:-/users/scro3801/repos/external/parcae}"
DATA_DIR="${DATA_DIR:-/datasets/$USER/parcae-data}"   # -> PARCAE_DATA in the configs
VENV="${VENV:-${PROJECT_DIR}/env_parcae}"
CONFIG="${CONFIG:-launch_configs/parcae-small-140m-rec1.yaml}"
GPUS="${GPUS:-4}"
# =============================================================================

set -x

RUN_NAME="${RUN_NAME:-$(basename "${CONFIG}" .yaml)}"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "=============================================="
echo "Job ${SLURM_JOB_ID} started $(date) on $(hostname)"
echo "Config:   ${CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "GPUs:     ${GPUS}"
echo "=============================================="

cd "${PROJECT_DIR}" || exit 1
nvidia-smi || true

# --- Environment ------------------------------------------------------------
source "${VENV}/bin/activate"
echo "python: $(which python)"

# The pinned lightning==2.3.0.dev20240328 calls pkg_resources.declare_namespace, which
# setuptools>=81 removed. Guarantee a compatible setuptools is present (idempotent, cached).
python -c "import pkg_resources" 2>/dev/null || pip install -q "setuptools<81"

# Config paths use ${PARCAE_DATA}; expandvars in settings.py resolves it.
export PARCAE_DATA="${DATA_DIR}"
# out_dir in the configs is an absolute path under /datasets/$USER/parcae-outputs -- the
# 50 GB NFS home cannot hold these checkpoints.
# Weights & Biases:
#   offline (default) -> logs to <out_dir>/wandb/, nothing sent to the web
#   online            -> streams to wandb.ai; needs auth (see below)
# Auth resolves from ~/.netrc (written by `wandb login`, on the shared /users volume so
# compute nodes see it) OR from WANDB_API_KEY. Only export the key when it is actually set:
# exporting an empty WANDB_API_KEY would OVERRIDE and break the ~/.netrc credential.
export WANDB_MODE="${WANDB_MODE:-offline}"
[ -n "${WANDB_API_KEY:-}" ] && export WANDB_API_KEY
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Fail fast and loudly rather than 30 minutes in.
[ -f "${PARCAE_DATA}/tokenizer/parcae_tokenizer/tokenizer.json" ] || {
    echo "FATAL: tokenizer missing at ${PARCAE_DATA}/tokenizer -- run download_data.slurm first"; exit 1; }
# The train split link is created by download_data.slurm; xlarge uses a different one, so
# only warn (do not hard-fail) if this specific path is absent.
[ -e "${PARCAE_DATA}/fineweb-edu-100b-shuffle/train" ] || \
    echo "WARNING: ${PARCAE_DATA}/fineweb-edu-100b-shuffle/train not found (ok for xlarge/350bt)"

# On preemption/timeout, tell train.py to force-save a checkpoint before we die.
# train.py installs a SIGUSR1 handler; forward the signal to the python child.
trap 'echo "Preemption signal received"; pkill -USR1 -f scripts/train.py 2>/dev/null || true; sleep 90' USR1

# devices is read as SLURM_NTASKS_PER_NODE -> LOCAL_WORLD_SIZE -> torch.cuda.device_count()
# (recpre/settings.py:313). We set --ntasks-per-node=1 for Slurm, but train.py must see the
# real per-node rank count or it derives the wrong gradient_accumulation_steps -- and that
# mistake still PASSES the world-batch assert, because `devices` sits on both sides of it.
# So pin it explicitly to the number of ranks torchrun will actually spawn.
export SLURM_NTASKS_PER_NODE="${GPUS}"
export SLURM_JOB_NUM_NODES=1

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${GPUS}" \
    --master_port="${MASTER_PORT}" \
    scripts/train.py \
    --config "${CONFIG}" \
    --run_name="${RUN_NAME}" \
    ${EXTRA_ARGS} &
TRAIN_PID=$!
wait "${TRAIN_PID}"
TRAIN_EXIT=$?

echo "=============================================="
echo "Training exited ${TRAIN_EXIT} at $(date)"
echo "=============================================="
exit $TRAIN_EXIT
