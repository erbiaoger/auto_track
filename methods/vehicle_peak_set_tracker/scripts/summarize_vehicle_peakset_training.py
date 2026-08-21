from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize vehicle peak-set training metrics.")
    parser.add_argument("--train-dir", required=True, type=Path, help="Training output directory with train_history.jsonl")
    parser.add_argument("--out-json", type=Path, default=None, help="Output JSON path; defaults to <train-dir>/training_summary.json")
    parser.add_argument("--out-md", type=Path, default=None, help="Output Markdown path; defaults to <train-dir>/training_summary.md")
    parser.add_argument("--prediction-summary", type=Path, default=None, help="Optional prediction summary.json to include")
    return parser.parse_args(argv)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty history: {path}")
    return rows


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except Exception:
        return None
    if val != val:
        return None
    return val


def _metrics(row: dict[str, Any], split: str) -> dict[str, float]:
    part = row.get(split, {})
    if not isinstance(part, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in part.items():
        f = _safe_float(value)
        if f is not None:
            out[str(key)] = f
    return out


def _match_ratio(metrics: dict[str, float]) -> float | None:
    gt = float(metrics.get("gt", 0.0))
    matched = float(metrics.get("matched", 0.0))
    if gt <= 0.0:
        return None
    return matched / gt


def _block_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    missing_t = metrics.get("loss_missing_complete_time")
    missing_v = metrics.get("loss_missing_complete_valid")
    dead_t = metrics.get("loss_dead_complete_time")
    dead_v = metrics.get("loss_dead_complete_valid")
    anchor = metrics.get("loss_anchor")
    anchor_time = metrics.get("loss_anchor_time")
    obj = metrics.get("loss_obj")
    count = metrics.get("loss_count")
    total_missing = None
    if missing_t is not None or missing_v is not None:
        total_missing = float((missing_t or 0.0) + (missing_v or 0.0))
    total_dead = None
    if dead_t is not None or dead_v is not None:
        total_dead = float((dead_t or 0.0) + (dead_v or 0.0))
    total_recognition = None
    if obj is not None or anchor is not None or anchor_time is not None or count is not None:
        total_recognition = float((obj or 0.0) + (anchor or 0.0) + (anchor_time or 0.0) + (count or 0.0))
    return {
        "loss_missing_total": total_missing,
        "loss_dead_total": total_dead,
        "loss_recognition_proxy": total_recognition,
        "match_ratio": _match_ratio(metrics),
    }


def _best_row(rows: list[dict[str, Any]], split: str, key: str) -> dict[str, Any] | None:
    best_row: dict[str, Any] | None = None
    best_value: float | None = None
    for row in rows:
        metrics = _metrics(row, split)
        value = metrics.get(key)
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_value = value
            best_row = row
    return best_row


def _format_float(value: float | None, precision: int = 6) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Vehicle Peak-Set Training Summary")
    lines.append("")
    lines.append(f"- Train dir: `{summary['train_dir']}`")
    lines.append(f"- History: `{summary['history_path']}`")
    lines.append(f"- Summary source: `{summary.get('source_summary_path', '-')}`")
    lines.append("")
    lines.append("## Best Epoch")
    best = summary["best"]
    lines.append(f"- Epoch: `{best['epoch']}`")
    lines.append(f"- Val loss: `{_format_float(best['val'].get('loss'))}`")
    lines.append(f"- Val match ratio: `{_format_float(best['val_derived'].get('match_ratio'))}`")
    lines.append(f"- Missing total: `{_format_float(best['val_derived'].get('loss_missing_total'))}`")
    lines.append(f"- Dead total: `{_format_float(best['val_derived'].get('loss_dead_total'))}`")
    lines.append(f"- Jump: `{_format_float(best['val'].get('loss_jump'))}`")
    lines.append(f"- Slope variation: `{_format_float(best['val'].get('loss_slope_variation'))}`")
    lines.append(f"- Recognition proxy: `{_format_float(best['val_derived'].get('loss_recognition_proxy'))}`")
    lines.append("")
    lines.append("## Last Epoch")
    last = summary["last"]
    lines.append(f"- Epoch: `{last['epoch']}`")
    lines.append(f"- Val loss: `{_format_float(last['val'].get('loss'))}`")
    lines.append(f"- Val match ratio: `{_format_float(last['val_derived'].get('match_ratio'))}`")
    lines.append(f"- Missing total: `{_format_float(last['val_derived'].get('loss_missing_total'))}`")
    lines.append(f"- Dead total: `{_format_float(last['val_derived'].get('loss_dead_total'))}`")
    lines.append(f"- Jump: `{_format_float(last['val'].get('loss_jump'))}`")
    lines.append(f"- Slope variation: `{_format_float(last['val'].get('loss_slope_variation'))}`")
    lines.append(f"- Recognition proxy: `{_format_float(last['val_derived'].get('loss_recognition_proxy'))}`")
    lines.append("")
    lines.append("## Key Validation Metrics")
    table_rows = [
        ("loss", "val loss"),
        ("loss_obj", "objectness"),
        ("loss_anchor", "anchor"),
        ("loss_anchor_time", "anchor_time"),
        ("loss_complete_time", "complete_time"),
        ("loss_complete_valid", "complete_valid"),
        ("loss_missing_complete_time", "missing_time"),
        ("loss_missing_complete_valid", "missing_valid"),
        ("loss_dead_complete_time", "dead_time"),
        ("loss_dead_complete_valid", "dead_valid"),
        ("loss_count", "count"),
        ("loss_jump", "jump"),
        ("loss_slope_variation", "slope_variation"),
        ("matched", "matched"),
        ("gt", "gt"),
    ]
    lines.append("| metric | best | last |")
    lines.append("| --- | ---: | ---: |")
    for key, label in table_rows:
        lines.append(
            f"| {label} | {_format_float(best['val'].get(key))} | {_format_float(last['val'].get(key))} |"
        )
    lines.append("")
    out_rows = summary.get("prediction_summary", {})
    if out_rows:
        lines.append("## Prediction Summary")
        for key in ("model", "dataset_dir", "plot_samples", "plot_style"):
            if key in out_rows:
                lines.append(f"- {key}: `{out_rows[key]}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train_dir = Path(args.train_dir).expanduser()
    history_path = train_dir / "train_history.jsonl"
    summary_path = train_dir / "summary.json"
    if not history_path.exists():
        raise FileNotFoundError(f"missing history: {history_path}")
    rows = _load_jsonl(history_path)
    last_row = rows[-1]
    best_row = _best_row(rows, "val", "loss")
    if best_row is None:
        best_row = last_row

    best = {
        "epoch": int(best_row["epoch"]),
        "train": _metrics(best_row, "train"),
        "val": _metrics(best_row, "val"),
    }
    best["train_derived"] = _block_metrics(best["train"])
    best["val_derived"] = _block_metrics(best["val"])

    last = {
        "epoch": int(last_row["epoch"]),
        "train": _metrics(last_row, "train"),
        "val": _metrics(last_row, "val"),
    }
    last["train_derived"] = _block_metrics(last["train"])
    last["val_derived"] = _block_metrics(last["val"])

    prediction_summary: dict[str, Any] | None = None
    if args.prediction_summary is not None:
        pred_path = Path(args.prediction_summary).expanduser()
        if pred_path.exists():
            prediction_summary = json.loads(pred_path.read_text(encoding="utf-8"))

    summary = {
        "train_dir": str(train_dir),
        "history_path": str(history_path),
        "source_summary_path": str(summary_path) if summary_path.exists() else None,
        "epoch_count": int(len(rows)),
        "best": best,
        "last": last,
        "prediction_summary": prediction_summary or {},
    }
    if summary_path.exists():
        try:
            summary["training_run_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary["training_run_summary"] = {}

    out_json = Path(args.out_json).expanduser() if args.out_json is not None else train_dir / "training_summary.json"
    out_md = Path(args.out_md).expanduser() if args.out_md is not None else train_dir / "training_summary.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(out_md, summary)

    print(json.dumps({"train_dir": str(train_dir), "out_json": str(out_json), "out_md": str(out_md), "best_epoch": best["epoch"], "last_epoch": last["epoch"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
