#!/bin/bash
# =============================================================================
# Export every finished stability arm to the Hugging Face two-file layout.
#
#   sbatch slurm/export_stab.slurm
#
# Each arm's Lightning checkpoint is a single pickle of the whole training
# state (weights, optimizer, dataloader position, RNG, FLOP accounting). It can
# only be unpickled with this repository importable, because it references
# ParcaeConfig -- including the stability-only `injection_fixed_decay` and
# `max_recurrence` fields -- and scripts/cost.py. scripts/export_hf.py pulls out
# just the weights and writes <arm>/export/{config.json,pytorch_model.bin},
# which loads with no training code and uploads to the Hub as-is.
#
# This is pure I/O, no GPU and almost no CPU: roughly 1.5 GB read and 650 MB
# written per arm, about ten minutes each, so ~2 h for twelve arms. It runs here
# rather than on the login node to keep that shared node's filesystem free.
# No --gres is requested; every partition on this cluster is GPU-attached, so
# this takes CPUs on a GPU node without occupying a GPU.
# =============================================================================
#SBATCH --job-name=parcae-export-stab
#SBATCH --account=mlrg_nsg
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/users/scro3801/repos/parcae-stability/slurm_logs/%x-%j.out
#SBATCH --error=/users/scro3801/repos/parcae-stability/slurm_logs/%x-%j.err

set -uo pipefail

REPO="${REPO:-/users/scro3801/repos/parcae-stability}"
STAB_DIR="${STAB_DIR:-/datasets/scro3801/parcae-outputs/stab}"
# Runs that are not sweep arms. The smoke test is 30 steps of plumbing check.
SKIP_ARMS="${SKIP_ARMS:-smoke}"

cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"
echo "python: $(which python)"
echo "arms under: $STAB_DIR"
echo "=============================================="

exported=0
skipped=0
failed=0

for arm_dir in "$STAB_DIR"/*/; do
    arm="$(basename "$arm_dir")"

    if [[ " $SKIP_ARMS " == *" $arm "* ]]; then
        echo "SKIP $arm (not a sweep arm)"
        skipped=$((skipped + 1))
        continue
    fi
    # TRAIN_DONE is written when an arm finishes; without it the newest
    # checkpoint is mid-training and must not be published.
    if [ ! -f "$arm_dir/TRAIN_DONE" ]; then
        echo "SKIP $arm (no TRAIN_DONE marker -- still training or failed)"
        skipped=$((skipped + 1))
        continue
    fi
    if [ ! -f "$arm_dir/model_config.json" ]; then
        echo "SKIP $arm (no model_config.json)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "---------- exporting $arm"
    if python scripts/export_hf.py "$arm_dir"; then
        exported=$((exported + 1))
    else
        echo "FAILED $arm"
        failed=$((failed + 1))
    fi
done

echo "=============================================="
echo "exported=$exported skipped=$skipped failed=$failed"
[ "$failed" -eq 0 ] || exit 1
