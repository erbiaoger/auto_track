#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python3}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/results/vehicle_peakset_run}"
DATASET_DIR="${DATASET_DIR:-${RUN_DIR}/peakguided_dataset_realshape_missing}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/peakguided_train_observed_aux_pairtail_noquad}"
RESUME="${RESUME:-${RUN_DIR}/peakguided_train_observed_aux_missing_noquad/checkpoint_last.pt}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-2e-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
SAVE_EVERY="${SAVE_EVERY:-5}"
LOG_EVERY="${LOG_EVERY:-20}"
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-1}"

OBSERVED_VALID_WEIGHT="${OBSERVED_VALID_WEIGHT:-3.0}"

if [[ ! -f "${RESUME}" ]]; then
  echo "missing RESUME checkpoint: ${RESUME}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_DIR}/meta.json" ]]; then
  echo "missing DATASET_DIR/meta.json: ${DATASET_DIR}/meta.json" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

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
  --no-residual-completion
  --freeze-except-observed-valid-and-pair-tail
  --reset-best-val
  --complete-time-weight 8.0
  --complete-valid-weight 1.0
  --observed-valid-weight "${OBSERVED_VALID_WEIGHT}"
  --missing-complete-time-weight 12.0
  --missing-complete-valid-weight 4.0
  --anchor-weight 1.0
  --anchor-time-weight 0.5
  --inertia-weight 0.80
  --linearity-weight 0.40
  --jump-weight 0.35
  --slope-variation-weight 0.50
)

if [[ "${AMP}" == "1" || "${AMP}" == "true" || "${AMP}" == "yes" ]]; then
  CMD+=(--amp)
fi

printf 'training_observed_aux_pairtail_noquad\n'
printf 'DATASET_DIR=%s\nOUT_DIR=%s\nRESUME=%s\nSTART_EPOCH=%s\nTOTAL_EPOCHS=%s\n' \
  "${DATASET_DIR}" "${OUT_DIR}" "${RESUME}" "${START_EPOCH}" "${TOTAL_EPOCHS}"
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
