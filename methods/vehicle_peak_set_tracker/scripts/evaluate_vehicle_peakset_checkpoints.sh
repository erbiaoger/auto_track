#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
TRAIN_DIR="${TRAIN_DIR:-$RUN_DIR/peakguided_train_realshape_easy_completion}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/test_dataset_realshape_missing}"
OUT_ROOT="${OUT_ROOT:-$RUN_DIR/eval_peakguided_easy_completion}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
PLOT_SAMPLES="${PLOT_SAMPLES:-0}"
PREDICT_PROFILE="${PREDICT_PROFILE:-recall}"

mkdir -p "$OUT_ROOT"

checkpoint_list=()
if [[ -n "${CHECKPOINTS:-}" ]]; then
  # shellcheck disable=SC2206
  checkpoint_list=($CHECKPOINTS)
else
  if [[ -f "$TRAIN_DIR/checkpoint_best.pt" ]]; then
    checkpoint_list+=("$TRAIN_DIR/checkpoint_best.pt")
  fi
  while IFS= read -r ckpt; do
    checkpoint_list+=("$ckpt")
  done < <(find "$TRAIN_DIR" -maxdepth 1 -type f -name 'checkpoint_epoch_*.pt' | sort)
fi

if [[ "${INCLUDE_BASELINE:-1}" == "1" ]]; then
  BASELINE="${BASELINE:-$RUN_DIR/peakguided_train_realshape_missingrefine/checkpoint_best.pt}"
  if [[ -f "$BASELINE" ]]; then
    checkpoint_list=("$BASELINE" "${checkpoint_list[@]}")
  fi
fi

if [[ "${INCLUDE_X2_REFS:-1}" == "1" ]]; then
  for ref in \
    "$RUN_DIR/peakguided_train_realshape_missingrefine_x2/checkpoint_epoch_225.pt" \
    "$RUN_DIR/peakguided_train_realshape_missingrefine_x2/checkpoint_epoch_235.pt"; do
    if [[ -f "$ref" ]]; then
      checkpoint_list+=("$ref")
    fi
  done
fi

summary_jsonl="$OUT_ROOT/checkpoint_metrics.jsonl"
: > "$summary_jsonl"

for ckpt in "${checkpoint_list[@]}"; do
  name="$(basename "$(dirname "$ckpt")")_$(basename "$ckpt" .pt)"
  pred_dir="$OUT_ROOT/$name"
  echo "evaluating $ckpt"
  MODEL="$ckpt" \
    DATASET_DIR="$DATASET_DIR" \
    OUT_DIR="$pred_dir" \
    NUM_SAMPLES="$NUM_SAMPLES" \
    PLOT_SAMPLES="$PLOT_SAMPLES" \
    PREDICT_PROFILE="$PREDICT_PROFILE" \
    ./scripts/predict_vehicle_peakset_test_realshape_cpu.sh >/dev/null

  "$PYTHON_BIN" scripts/evaluate_vehicle_peakset_predictions.py \
    --dataset-dir "$DATASET_DIR" \
    --predictions "$pred_dir/predictions.jsonl" \
    --out-json "$pred_dir/eval_metrics.json" >/dev/null

  "$PYTHON_BIN" - "$pred_dir/eval_metrics.json" "$ckpt" >> "$summary_jsonl" <<'PY'
import json, pathlib, sys
metrics = json.loads(pathlib.Path(sys.argv[1]).read_text())
metrics["checkpoint"] = sys.argv[2]
print(json.dumps(metrics, ensure_ascii=False))
PY
done

"$PYTHON_BIN" - "$summary_jsonl" "$OUT_ROOT/checkpoint_metrics_summary.json" <<'PY'
import json, pathlib, sys
rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.strip()]
def score(row):
    recall = float(row.get("recall") or 0.0)
    precision = float(row.get("precision") or 0.0)
    missing = row.get("missing_completion_mae_s")
    jump = float(row.get("jump_rate") or 0.0)
    zero = float(row.get("zero_track_samples") or 0.0)
    missing_penalty = 0.0 if missing is None else min(1.0, float(missing) / 6.0)
    return recall * 2.0 + precision - missing_penalty - jump - zero * 0.05
for row in rows:
    row["selection_score"] = score(row)
rows.sort(key=lambda item: item["selection_score"], reverse=True)
payload = {"best": rows[0] if rows else None, "checkpoints": rows}
pathlib.Path(sys.argv[2]).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
if rows:
    best = rows[0]
    print(json.dumps({k: best.get(k) for k in ["checkpoint", "selection_score", "recall", "precision", "missing_completion_mae_s", "avg_track_count", "zero_track_samples", "jump_rate"]}, ensure_ascii=False))
PY
