#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python3}"

DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/results/vehicle_peakset_run/peakguided_dataset_residual_completion}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/results/vehicle_peakset_run/peakguided_train_observed_completion}"
RESUME="${RESUME:-${ROOT_DIR}/results/vehicle_peakset_run/peakguided_train_residual_completion/checkpoint_last.pt}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-20}"
LR="${LR:-5e-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SAVE_EVERY="${SAVE_EVERY:-5}"
LOG_EVERY="${LOG_EVERY:-10}"
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-1}"

COMPLETE_TIME_WEIGHT="${COMPLETE_TIME_WEIGHT:-8.0}"
COMPLETE_VALID_WEIGHT="${COMPLETE_VALID_WEIGHT:-1.0}"
OBSERVED_VALID_WEIGHT="${OBSERVED_VALID_WEIGHT:-2.0}"
MISSING_COMPLETE_TIME_WEIGHT="${MISSING_COMPLETE_TIME_WEIGHT:-12.0}"
MISSING_COMPLETE_VALID_WEIGHT="${MISSING_COMPLETE_VALID_WEIGHT:-3.0}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-1.0}"
ANCHOR_TIME_WEIGHT="${ANCHOR_TIME_WEIGHT:-0.75}"
INERTIA_WEIGHT="${INERTIA_WEIGHT:-0.50}"
LINEARITY_WEIGHT="${LINEARITY_WEIGHT:-0.35}"
JUMP_WEIGHT="${JUMP_WEIGHT:-0.20}"
SLOPE_VARIATION_WEIGHT="${SLOPE_VARIATION_WEIGHT:-0.30}"

MAX_RESIDUAL_NORM="${MAX_RESIDUAL_NORM:-0.005}"
MAX_BASE_SLOPE_NORM="${MAX_BASE_SLOPE_NORM:-0.75}"
MAX_BASE_CURVE_NORM="${MAX_BASE_CURVE_NORM:-0.35}"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${RESUME}" ]]; then
  echo "missing RESUME checkpoint: ${RESUME}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "missing DATASET_DIR: ${DATASET_DIR}" >&2
  exit 1
fi

START_EPOCH="$("${PYTHON_BIN}" - "${RESUME}" <<'PY'
import sys
import torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ckpt.get("epoch", 0)) + 1)
PY
)"
TOTAL_EPOCHS=$((START_EPOCH + FINETUNE_EPOCHS - 1))

CMD=(
  "${PYTHON_BIN}" -m autotrack.dl.train_vehicle_peak_set
  --dataset-dir "${DATASET_DIR}"
  --out-dir "${OUT_DIR}"
  --resume "${RESUME}"
  --device "${DEVICE}"
  --epochs "${TOTAL_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --shuffle
  --save-every "${SAVE_EVERY}"
  --log-every "${LOG_EVERY}"
  --complete-time-weight "${COMPLETE_TIME_WEIGHT}"
  --complete-valid-weight "${COMPLETE_VALID_WEIGHT}"
  --observed-valid-weight "${OBSERVED_VALID_WEIGHT}"
  --missing-complete-time-weight "${MISSING_COMPLETE_TIME_WEIGHT}"
  --missing-complete-valid-weight "${MISSING_COMPLETE_VALID_WEIGHT}"
  --anchor-weight "${ANCHOR_WEIGHT}"
  --anchor-time-weight "${ANCHOR_TIME_WEIGHT}"
  --inertia-weight "${INERTIA_WEIGHT}"
  --linearity-weight "${LINEARITY_WEIGHT}"
  --jump-weight "${JUMP_WEIGHT}"
  --slope-variation-weight "${SLOPE_VARIATION_WEIGHT}"
  --residual-completion
  --max-residual-norm "${MAX_RESIDUAL_NORM}"
  --max-base-slope-norm "${MAX_BASE_SLOPE_NORM}"
  --max-base-curve-norm "${MAX_BASE_CURVE_NORM}"
)

if [[ "${AMP}" == "1" || "${AMP}" == "true" || "${AMP}" == "yes" ]]; then
  CMD+=(--amp)
fi

printf 'training_observed_completion\n'
printf 'DATASET_DIR=%s\nOUT_DIR=%s\nRESUME=%s\nSTART_EPOCH=%s\nTOTAL_EPOCHS=%s\n' \
  "${DATASET_DIR}" "${OUT_DIR}" "${RESUME}" "${START_EPOCH}" "${TOTAL_EPOCHS}"
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
