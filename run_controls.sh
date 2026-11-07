#!/usr/bin/env bash
# =============================================================================
# Submit the whole recurrence-ablated control pipeline:
#
#     1. one CPU-only job that downloads the tokenizer + dataset
#     2. one 4-GPU training job per model size, each chained to run ONLY IF
#        the download finished successfully (--dependency=afterok)
#
# Usage:
#     ./run_controls.sh                # submit everything
#     ./run_controls.sh --dry-run      # print the sbatch commands, submit nothing
#     ./run_controls.sh --skip-download   # data already present; submit training only
#
# Everything you might want to change is in the CONFIGURATION block below.
# =============================================================================
set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Where things live -------------------------------------------------------
# The dataset (~179 GB) cannot live on your 50 GB home volume, so it stays on
# /datasets (20 TB). Checkpoints are repo-local, which only fits because
# keep_last_n_checkpoints=1 caps each run at one checkpoint -- see the disk table
# printed below. Both are bind-mounted into the container, which is what makes
# the /resource/... paths already baked into the YAML configs resolve correctly.
PROJECT_DIR="/users/scro3801/repos/external/parcae"   # this repo         -> /resource
DATA_DIR="/datasets/$USER/parcae-data"                # dataset+tokenizer -> /resource/data
# Checkpoints go on /datasets, NOT the 50 GB NFS home: three runs hold ~17 GB resident and
# peak at ~27 GB while a new checkpoint lands before the old one is pruned, against ~35 GB
# free on home. This must match the absolute out_dir in each launch_configs/*-rec1.yaml.
OUT_DIR="/datasets/$USER/parcae-outputs"              # checkpoints

# Slurm writes job logs here (as files inside these dirs, one per job).
SLURM_OUT_DIR="${PROJECT_DIR}/slurm_out"
SLURM_ERR_DIR="${PROJECT_DIR}/slurm_err"

# --- Slurm -------------------------------------------------------------------
ACCOUNT="mlrg_nsg"          # your only account; Slurm rejects jobs without it
# a100 | b200 ONLY. Do NOT use h200: that partition is eight separate 1-GPU nodes
# (14 CPU, 160 GB each), so launch_control.slurm's --nodes=1 --gres=gpu:4 --cpus-per-task=32
# --mem=200000 can never be scheduled there and the job pends forever. As of 2026-09-18
# civo-a100-1 (8x A100-80GB) is completely idle with an empty queue, while civo-b200-1 is
# 8/8 allocated with four jobs queued ahead. B200 is ~5x faster when you can get it.
GPU_PARTITION="a100"
CPU_PARTITION="a100"        # download job requests NO GPUs, so it blocks nobody
# 4 is the hard maximum: the a100/h200/b200 QOS each cap MaxTRESPerUser at gres/gpu=4.
GPUS=4                      # GPUs per training job (node has 8, so two jobs can share)

# --- What to run -------------------------------------------------------------
# Runtimes at ~35-40% MFU (4 GPUs):   4xA100     4xB200
#                            small      ~6 h       ~1 h
#                            medium    ~43 h       ~7 h
#                            large     ~7.5 d     ~30 h
#                            xlarge    ~21 d      ~3.6 d
# Anything over the 2-day partition limit is requeued mid-run. xlarge additionally
# needs the 937 GB 350BT dataset rather than the 179 GB 100BT one: at this repo's
# ~78 tokens/param recipe its 104B-token budget does not fit a 100B-token corpus.
SIZES=(small-140m medium-370m large-770m)   # append xlarge-1_3b to include the 1.3B control

# --- Environment (native, no Docker) -----------------------------------------
# This cluster has no container runtime, so jobs run the repo's venv directly.
VENV="${PROJECT_DIR}/env_parcae"

# --- Weights & Biases --------------------------------------------------------
# Default online
WANDB_MODE="${WANDB_MODE:-online}"

# --- Extra args appended to every train.py invocation ------------------------
EXTRA_ARGS=""

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

DRY_RUN=false
SKIP_DOWNLOAD=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)       DRY_RUN=true ;;
        --skip-download) SKIP_DOWNLOAD=true ;;
        -h|--help)       sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 1 ;;
    esac
done

# Which dataset to fetch is implied by SIZES: xlarge reads the 350BT split, the
# rest read 100BT. Sizes measured from the HF API for SandyResearch/fineweb-edu-shuffled.
needs_100bt=false; needs_350bt=false
for s in "${SIZES[@]}"; do
    if [[ "$s" == "xlarge-1_3b" ]]; then needs_350bt=true; else needs_100bt=true; fi
done
if   $needs_100bt && $needs_350bt; then DATASET="fineweb";       DATA_GB=1110
elif $needs_350bt;                 then DATASET="fineweb-350bt"; DATA_GB=937
else                                    DATASET="fineweb-100bt"; DATA_GB=179
fi

# --- Preflight ---------------------------------------------------------------
cd "$PROJECT_DIR"
fail=0
for f in download_data.slurm launch_control.slurm; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f in $PROJECT_DIR" >&2; fail=1; }
done
[[ -f "$VENV/bin/activate" ]] || { echo "ERROR: venv not found at $VENV" >&2; fail=1; }
for s in "${SIZES[@]}"; do
    [[ -f "launch_configs/parcae-$s-rec1.yaml" ]] || {
        echo "ERROR: missing launch_configs/parcae-$s-rec1.yaml" >&2; fail=1; }
done
mkdir -p "$DATA_DIR" "$OUT_DIR" "$SLURM_OUT_DIR" "$SLURM_ERR_DIR" || fail=1
(( fail == 0 )) || exit 1

# Checkpoint disk. With keep_last_n_checkpoints=1 each run holds one checkpoint, but pruning
# happens after the new save lands, so peak is (all runs' resident) + (largest single ckpt).
declare -A CKPT_GB=([small-140m]=2 [medium-370m]=5 [large-770m]=10 [xlarge-1_3b]=16)
ckpt_resident=0; ckpt_largest=0
for s in "${SIZES[@]}"; do
    g=${CKPT_GB[$s]:-16}
    ckpt_resident=$(( ckpt_resident + g ))
    (( g > ckpt_largest )) && ckpt_largest=$g
done
ckpt_peak=$(( ckpt_resident + ckpt_largest ))
out_avail_gb=$(df -BG --output=avail "$OUT_DIR" | tail -1 | tr -dc '0-9')

avail_gb=$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc '0-9')
echo "=============================================================="
echo " Parcae recurrence-1 control pipeline"
echo "=============================================================="
printf ' %-18s %s\n' \
    "project"    "$PROJECT_DIR" \
    "data dir"   "$DATA_DIR" \
    "output dir" "$OUT_DIR" \
    "account"    "$ACCOUNT" \
    "gpu part."  "$GPU_PARTITION ($GPUS GPUs/job)" \
    "cpu part."  "$CPU_PARTITION (download, 0 GPUs)" \
    "sizes"      "${SIZES[*]}" \
    "dataset"    "$DATASET (~${DATA_GB} GB)" \
    "wandb"      "$WANDB_MODE" \
    "data vol free" "${avail_gb} GB (needs ~${DATA_GB} GB)" \
    "ckpt vol free" "${out_avail_gb} GB (needs ~${ckpt_resident} GB, ~${ckpt_peak} GB at peak)"
echo "=============================================================="

if (( avail_gb < DATA_GB )); then
    echo "WARNING: only ${avail_gb} GB free on $DATA_DIR but ${DATASET} needs ~${DATA_GB} GB." >&2
    echo "         (the download may also keep two copies -- see the job's du -sh output)" >&2
fi
if (( out_avail_gb < ckpt_peak )); then
    echo "WARNING: only ${out_avail_gb} GB free on $OUT_DIR but checkpoints peak at ~${ckpt_peak} GB." >&2
    echo "         Point OUT_DIR at /datasets/\$USER/parcae-outputs instead." >&2
fi

# Exported so sbatch (--export=ALL by default) hands them to the job scripts.
export PROJECT_DIR DATA_DIR OUT_DIR VENV DATASET GPUS WANDB_MODE EXTRA_ARGS

run() {  # echo the command; run it unless --dry-run
    if $DRY_RUN; then echo "  [dry-run] $*" >&2; echo "DRYRUN"; else "$@"; fi
}

# --- 1. Download -------------------------------------------------------------
DEP_ARG=()
if $SKIP_DOWNLOAD; then
    echo ">> Skipping download (--skip-download); training jobs will start immediately."
else
    DL_JOB=$(run sbatch --parsable \
        --account="$ACCOUNT" --partition="$CPU_PARTITION" \
        --job-name="parcae-dl" \
        --output="${SLURM_OUT_DIR}/%x-%j.out" --error="${SLURM_ERR_DIR}/%x-%j.err" \
        download_data.slurm)
    echo ">> Download job: $DL_JOB   (${DATASET}, ~${DATA_GB} GB + tokenizer)"
    DEP_ARG=(--dependency="afterok:${DL_JOB}")
fi

# --- 2. Training -------------------------------------------------------------
declare -a SUBMITTED=()
for s in "${SIZES[@]}"; do
    cfg="launch_configs/parcae-${s}-rec1.yaml"
    jid=$(CONFIG="$cfg" run sbatch --parsable \
        --account="$ACCOUNT" --partition="$GPU_PARTITION" \
        --gres="gpu:${GPUS}" \
        --job-name="rec1-${s}" \
        --output="${SLURM_OUT_DIR}/%x-%j.out" --error="${SLURM_ERR_DIR}/%x-%j.err" \
        "${DEP_ARG[@]}" \
        launch_control.slurm)
    SUBMITTED+=("$jid  parcae-${s}-rec1")
    echo ">> Training job: $jid  <- $cfg"
done

echo "=============================================================="
if $DRY_RUN; then
    echo " Dry run: nothing was submitted."
else
    printf ' %s\n' "${SUBMITTED[@]}"
    echo
    echo " Training jobs stay PENDING (Dependency) until the download exits 0."
    echo
    echo " Watch:    squeue -u \$USER"
    echo " Logs:     tail -f ${SLURM_OUT_DIR}/rec1-small-140m-<jobid>.out"
    echo " Errors:   tail -f ${SLURM_ERR_DIR}/rec1-small-140m-<jobid>.err"
    echo " Ckpts:    ls -lh ${OUT_DIR}/*/checkpoints-DDPStrategy/"
    echo " Cancel:   scancel -u \$USER --name=rec1-small-140m"
fi
echo "=============================================================="
