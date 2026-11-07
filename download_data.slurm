#!/bin/bash
# =============================================================================
# CPU-only data + tokenizer download for the Parcae control runs.
# Native (no Docker): this cluster has no container runtime, so we use the
# repo's env_parcae venv directly. Requests no GPUs, so it blocks nobody.
#
#   sbatch download_data.slurm                         # fineweb-100bt (~179 GB)
#   DATASET=fineweb-350bt sbatch download_data.slurm   # (~937 GB, xlarge only)
#
# Sizes (measured from the HF API, SandyResearch/fineweb-edu-shuffled):
#   sample-100BT  1823 shards  171 GB   -> small / medium / large
#   sample-350BT  1338 shards  929 GB   -> xlarge only
#   val            218 shards  7.3 GB   -> always
# =============================================================================
#SBATCH --job-name=parcae-download
#SBATCH --account=mlrg_nsg
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=12:00:00
# Paths are relative to the directory you run sbatch from (run_controls.sh cds to the repo
# first, and also passes absolute paths that override these). %x = job name, %j = job id.
# Slurm does NOT create these directories -- they must exist at submit time.
#SBATCH --output=./slurm_out/%x-%j.out
#SBATCH --error=./slurm_err/%x-%j.err

# =============================================================================
# User configuration
# =============================================================================
PROJECT_DIR="${PROJECT_DIR:-/users/scro3801/repos/external/parcae}"
DATA_DIR="${DATA_DIR:-/datasets/$USER/parcae-data}"   # -> PARCAE_DATA in the configs
VENV="${VENV:-${PROJECT_DIR}/env_parcae}"
DATASET="${DATASET:-fineweb-100bt}"                   # or fineweb-350bt / fineweb / fineweb-val
# =============================================================================

set -x
set -euo pipefail

echo "=============================================="
echo "Download job ${SLURM_JOB_ID} started $(date) on $(hostname)"
echo "Dataset:   ${DATASET}"
echo "DATA_DIR:  ${DATA_DIR}"
echo "=============================================="

mkdir -p "${DATA_DIR}"
cd "${PROJECT_DIR}"
df -h "${DATA_DIR}" || true

# --- Environment ------------------------------------------------------------
source "${VENV}/bin/activate"
echo "python: $(which python)"
# HF downloads go straight to DATA_DIR; keep the hub cache there too so we don't
# fill the 50 GB home volume.
export HF_HOME="${DATA_DIR}/.hf_home"
export HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "${HF_HOME}"

# --- 1. Tokenizer -----------------------------------------------------------
# Configs expect ${PARCAE_DATA}/tokenizer/parcae_tokenizer/tokenizer.json.
# It is NOT in the repo and NOT in the model repos - it lives in its own HF repo.
echo "--- Downloading tokenizer ---"
python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='SandyResearch/parcae-tokenizer',
                      local_dir='${DATA_DIR}/tokenizer/parcae_tokenizer')
print('tokenizer ->', p)
"

# --- 2. Dataset -------------------------------------------------------------
echo "--- Downloading ${DATASET} ---"
DATA_DIR="${DATA_DIR}" python scripts/download_data.py "${DATASET}"

# --- 3. Reconcile paths -----------------------------------------------------
# download_data.py writes to  $DATA_DIR/fineweb-edu-shuffled/<split>/data/<split>/*.parquet
# but every launch config reads  ${PARCAE_DATA}/fineweb-edu-100b-shuffle/{train,val,val-resharded}
# and                           ${PARCAE_DATA}/fineweb-edu-350b-fixed-train
# Those names come from the authors' own cluster and are never created by the script, so we
# symlink the expected names onto the real ones. This leaves all YAML configs untouched.
echo "--- Linking expected config paths ---"
BASE="${DATA_DIR}/fineweb-edu-shuffled"
link() {  # link <link_path> <target_dir>
    if [ -d "$2" ]; then
        mkdir -p "$(dirname "$1")"
        ln -sfnT "$2" "$1"
        echo "linked $1 -> $2 ($(ls "$2" | wc -l) files)"
    else
        echo "skip $1: target $2 not present"
    fi
}
link "${DATA_DIR}/fineweb-edu-100b-shuffle/train"         "$BASE/sample-100BT/data/sample-100BT"
link "${DATA_DIR}/fineweb-edu-100b-shuffle/val"           "$BASE/val/data/val"
link "${DATA_DIR}/fineweb-edu-100b-shuffle/val-resharded" "$BASE/val/data/val"
link "${DATA_DIR}/fineweb-edu-350b-fixed-train"           "$BASE/sample-350BT/data/sample-350BT"

# --- 4. Verify --------------------------------------------------------------
echo "--- Verification ---"
ok=0
for p in "${DATA_DIR}/fineweb-edu-100b-shuffle/train" \
         "${DATA_DIR}/fineweb-edu-100b-shuffle/val" \
         "${DATA_DIR}/fineweb-edu-100b-shuffle/val-resharded" \
         "${DATA_DIR}/fineweb-edu-350b-fixed-train"; do
    if [ -d "$p" ]; then
        n=$(find -L "$p" -name "*.parquet" | wc -l)
        echo "  $p: $n parquet shards"
        [ "$n" -gt 0 ] || { echo "  ERROR: no shards in $p"; ok=1; }
    else
        echo "  $p: absent (fine if this split was not requested)"
    fi
done
if [ -f "${DATA_DIR}/tokenizer/parcae_tokenizer/tokenizer.json" ]; then
    echo "  tokenizer.json: OK"
else
    echo "  ERROR: tokenizer.json missing"; ok=1
fi

du -sh "${DATA_DIR}" || true
df -h "${DATA_DIR}" || true

if [ "$ok" -ne 0 ]; then
    echo "Download verification FAILED"; exit 1
fi
echo "=============================================="
echo "Download job finished OK at $(date)"
echo "=============================================="
