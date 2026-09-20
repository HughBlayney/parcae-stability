#!/bin/bash
# =============================================================================
# Export named training runs to the Hugging Face two-file layout.
#
#   RUNS="/datasets/$USER/parcae-outputs/parcae-small-140m-rec1" \
#       sbatch slurm/export_runs.slurm
#
# Companion to export_stab.slurm, which sweeps a whole directory; this takes an
# explicit list, so it suits the recurrence-ablated controls that finish one at
# a time. Same output per run: <run>/export/{config.json,pytorch_model.bin}.
#
# Run this rather than exporting on the login node. Measured 2026-09-20: the
# twelve-arm sweep (24 GB read) took 14 minutes here, while a single 1.5 GB
# control checkpoint had not finished after 27 minutes on the login node, whose
# NFS path is far slower under contention.
#
# REPO supplies export_hf.py and the parcae_lm/cost classes the checkpoint
# pickle references. The default works for both the sweep and the controls,
# because this fork's ParcaeConfig is a superset of the one they were saved
# with. Override REPO and VENV to unpickle with the exact training tree instead.
#
# No --gres: every partition here is GPU-attached, so this takes CPUs on a GPU
# node without occupying a GPU.
# =============================================================================
#SBATCH --job-name=parcae-export-runs
#SBATCH --account=mlrg_nsg
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/users/scro3801/repos/parcae-stability/slurm_logs/%x-%j.out
#SBATCH --error=/users/scro3801/repos/parcae-stability/slurm_logs/%x-%j.err

set -uo pipefail

REPO="${REPO:-/users/scro3801/repos/parcae-stability}"
VENV="${VENV:-$REPO/.venv}"
RUNS="${RUNS:?Set RUNS to one or more run directories, space separated}"

cd "$REPO" || exit 1
source "$VENV/bin/activate"
echo "python: $(which python)"
echo "repo:   $REPO"
echo "=============================================="

exported=0
failed=0

for run in $RUNS; do
    if [ ! -f "$run/model_config.json" ]; then
        echo "FAILED $run (no model_config.json)"
        failed=$((failed + 1))
        continue
    fi
    # Refuse to publish a checkpoint from a run that is still training: the
    # newest checkpoint would be mid-run, and rotation may delete it later.
    if squeue -h -u "$USER" -o "%Z" 2>/dev/null | grep -qx "$run"; then
        echo "FAILED $run (a job is still writing here)"
        failed=$((failed + 1))
        continue
    fi

    echo "---------- exporting $(basename "$run")"
    if python scripts/export_hf.py "$run"; then
        exported=$((exported + 1))
    else
        echo "FAILED $run"
        failed=$((failed + 1))
    fi
done

echo "=============================================="
echo "exported=$exported failed=$failed"
[ "$failed" -eq 0 ] || exit 1
