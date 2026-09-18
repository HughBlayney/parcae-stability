#!/bin/bash
# Stability-extrapolation sweep (see stability_extrapolation_plan.md).
#
# Usage:
#   bash runs/sweep_stability.sh <GPUS> [SCHEMES] [RHOS]
#   e.g. bash runs/sweep_stability.sh 4 "poisson-truncated-full" "learned 0.25 1.0 1.25"   # headline four
#        bash runs/sweep_stability.sh 4 "fixed" "learned 0.25 0.5 0.75 1.0 1.25"
#
# Environment overrides:
#   CONFIG        launch config            (default launch_configs/parcae-small-140m-stab.yaml)
#   BASE_OUT      output root              (default outputs/stab)
#   VAL_DIR       directory of val parquet shards for the held-out evals   (REQUIRED)
#   TOK           tokenizer dir override for scripts/check_state_norm.py   (default: from run_config.json)
#   DEPTHS        depths for the single-trajectory depth curve             (default 1..64 list below)
#   EVAL_T_LIST   depths to cross-check with scripts/eval.py               (default "8 16 64"; "" to skip)
#   EVAL_DOCS     documents for the held-out evals                         (default 1000)
#   EXTRA_TRAIN   extra CLI args for scripts/train.py (e.g. "--max_tokens 0 --max_steps 30" for a smoke run)
#   EXTRA_ARM     extra per-arm model overrides (e.g. "--model_overwrite.state_init zero")
#   TORCHRUN      launcher (default "torchrun --standalone --nproc_per_node=$GPUS")
#   PYTHON        python for single-process scripts (default python)
#
# Per arm it (1) trains (resuming if a partial checkpoint exists, skipping if TRAIN_DONE exists),
# (2) runs scripts/check_state_norm.py once (loss/PPL + state norms at every depth from one trajectory),
# (3) cross-checks a few depths with scripts/eval.py --override_mean_recurrence.
set -euo pipefail

GPUS="${1:-4}"
SCHEMES="${2:-poisson-truncated-full fixed}"
RHOS="${3:-learned 0.25 0.5 0.75 1.0 1.25}"
CONFIG="${CONFIG:-launch_configs/parcae-small-140m-stab.yaml}"
BASE_OUT="${BASE_OUT:-outputs/stab}"
VAL_DIR="${VAL_DIR:?set VAL_DIR to the directory containing the validation parquet shards}"
DEPTHS="${DEPTHS:-1,2,3,4,6,8,10,12,16,20,24,32,48,64}"
EVAL_T_LIST="${EVAL_T_LIST-8 16 64}"
EVAL_DOCS="${EVAL_DOCS:-1000}"
EXTRA_TRAIN="${EXTRA_TRAIN:-}"
EXTRA_ARM="${EXTRA_ARM:-}"
TORCHRUN="${TORCHRUN:-torchrun --standalone --nproc_per_node=$GPUS}"
PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:-.}"

for scheme in $SCHEMES; do
  for rho in $RHOS; do
    tag="${scheme%%-*}_rho${rho}"            # poisson_rho0.5, fixed_rholearned, ...
    OUT="${BASE_OUT}/${tag}"
    RUN_NAME="stab-${tag}"
    ARM_ARGS="--model_overwrite.sampling_scheme $scheme $EXTRA_ARM"
    [[ "$rho" != "learned" ]] && ARM_ARGS="$ARM_ARGS --model_overwrite.injection_fixed_decay $rho"
    mkdir -p "$OUT"

    # ---------------- training ----------------
    if [[ -f "$OUT/TRAIN_DONE" ]]; then
      echo "[$tag] training complete, skipping"
    else
      RESUME=false
      ls "$OUT"/checkpoints-*/step-* &>/dev/null && RESUME=true
      echo "[$tag] training (resume=$RESUME) -> $OUT"
      $TORCHRUN scripts/train.py \
        --config "$CONFIG" --out_dir "$OUT" --run_name "$RUN_NAME" --resume $RESUME \
        $ARM_ARGS $EXTRA_TRAIN \
        --logger_group stability-extrapolation 2>&1 | tee -a "$OUT/train.log"
      touch "$OUT/TRAIN_DONE"
    fi

    # ---------------- held-out depth curve (single trajectory, all depths) ----------------
    if [[ -f "$OUT/eval/depth_curve.json" ]]; then
      echo "[$tag] depth curve exists, skipping"
    else
      echo "[$tag] depth curve"
      $PYTHON scripts/check_state_norm.py --out_dir "$OUT" --val_data_dir "$VAL_DIR" \
        --depths "$DEPTHS" --max_samples "$EVAL_DOCS" ${TOK:+--tokenizer_path "$TOK"} \
        --output "$OUT/eval/depth_curve.json" 2>&1 | tee "$OUT/depth_curve.log"
    fi

    # ---------------- cross-check with scripts/eval.py ----------------
    for T in $EVAL_T_LIST; do
      if [[ -f "$OUT/eval/val_loss_T${T}.json" ]]; then
        echo "[$tag] eval.py T=$T exists, skipping"; continue
      fi
      echo "[$tag] eval.py T=$T"
      $TORCHRUN scripts/eval.py \
        --out_dir "$OUT" --config eval_configs/eval-val-loss.yaml --eval_name "val_loss_T${T}" \
        --tasks.bpb.val_data_dir "$VAL_DIR" --tasks.bpb.max_samples "$EVAL_DOCS" \
        --override_mean_recurrence "$T" 2>&1 | tee "$OUT/eval_T${T}.log"
    done
  done
done
echo "Sweep complete: $BASE_OUT"
