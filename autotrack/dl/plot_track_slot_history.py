"""Plot TrackSlotNet training history curves.

Purpose:
    Read `train_history.jsonl` written by `train_track_slot.py` and plot epoch
    curves for metrics such as loss, F1, count error, objectness, and timing.
    The script is intended for quickly judging whether a TrackSlotNet run is
    converging and whether validation metrics follow training metrics.

Example:
    uv run python -m autotrack.dl.plot_track_slot_history \
        --run-dir models/track_slot_cuda \
        --metrics loss,track_f1,count_mae,time_mae_norm \
        --out-dir models/track_slot_cuda/history_plots \
        --separate

Arguments:
    --run-dir points to a training output directory containing
    `train_history.jsonl`.
    --history can be used instead to pass the JSONL file directly.
    --metrics is a comma-separated list of base metric names. The script will
    draw `train_<metric>` and `val_<metric>` curves when those fields exist.
    --list-metrics prints available metric names and exits without plotting.

Outputs:
    <out-dir>/history_overview.png
        Multi-panel plot for the selected metrics.
    <out-dir>/history_<metric>.png
        One figure per metric when `--separate` is set.
    <out-dir>/history_plot_summary.json
        Available metrics, selected metrics, latest values, and best values.

Notes:
    Plot fonts are configured as Times New Roman for consistency with the
    project plotting requirement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_METRICS = "loss,track_f1,count_mae,time_mae_norm,max_objectness,mean_objectness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TrackSlotNet train_history.jsonl curves.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Training output directory containing train_history.jsonl.")
    parser.add_argument("--history", type=Path, default=None, help="Path to train_history.jsonl; overrides --run-dir.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for plots; defaults to <run-dir>/history_plots.")
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="Comma-separated base metric names to plot, or 'auto'.")
    parser.add_argument("--prefixes", default="train,val", help="Comma-separated prefixes to plot, usually train,val.")
    parser.add_argument("--smooth-window", type=int, default=1, help="Moving-average window in epochs; 1 disables smoothing.")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Figure output format.")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI for PNG output.")
    parser.add_argument("--title", default="", help="Optional figure title.")
    parser.add_argument("--separate", action="store_true", help="Also write one figure per metric.")
    parser.add_argument("--list-metrics", action="store_true", help="Print available base metric names and exit.")
    return parser.parse_args()


def _resolve_history_path(args: argparse.Namespace) -> Path:
    if args.history is not None:
        return Path(args.history).expanduser()
    if args.run_dir is None:
        return Path("models/track_slot_cuda/train_history.jsonl").expanduser()
    return Path(args.run_dir).expanduser() / "train_history.jsonl"


def _resolve_out_dir(args: argparse.Namespace, history_path: Path) -> Path:
    if args.out_dir is not None:
        return Path(args.out_dir).expanduser()
    if args.run_dir is not None:
        return Path(args.run_dir).expanduser() / "history_plots"
    return history_path.parent / "history_plots"


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"History file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if "epoch" not in row:
                raise ValueError(f"Missing epoch at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"History file is empty: {path}")
    rows.sort(key=lambda item: float(item.get("epoch", 0.0)))
    return rows


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _available_base_metrics(rows: list[dict[str, Any]], prefixes: list[str]) -> list[str]:
    bases: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if not _is_number(value):
                continue
            for prefix in prefixes:
                stem = f"{prefix}_"
                if key.startswith(stem):
                    base = key[len(stem) :]
                    if base not in {"epoch"}:
                        bases.add(base)
    return sorted(bases)


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_series(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key)
        if _is_number(value) and _is_number(row.get("epoch")):
            xs.append(float(row["epoch"]))
            ys.append(float(value))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    w = int(max(1, window))
    if w <= 1 or y.size < w:
        return y
    kernel = np.ones(w, dtype=np.float64) / float(w)
    left = w // 2
    right = w - 1 - left
    padded = np.pad(y, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )
    return plt


def _plot_metric(
    ax: Any,
    rows: list[dict[str, Any]],
    metric: str,
    prefixes: list[str],
    *,
    smooth_window: int,
) -> bool:
    plotted = False
    colors = {"train": "#1f77b4", "val": "#d62728"}
    for prefix in prefixes:
        key = f"{prefix}_{metric}"
        x, y = _metric_series(rows, key)
        if x.size == 0:
            continue
        ax.plot(x, _smooth(y, smooth_window), marker="o", markersize=3, linewidth=1.6, label=prefix, color=colors.get(prefix))
        plotted = True
    ax.set_title(metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(frameon=True)
    return plotted


def _select_metrics(rows: list[dict[str, Any]], prefixes: list[str], requested: str) -> list[str]:
    available = _available_base_metrics(rows, prefixes)
    if str(requested).strip().lower() == "auto":
        preferred = _split_csv(DEFAULT_METRICS)
        selected = [metric for metric in preferred if metric in available]
        return selected or available[:8]
    selected = [metric for metric in _split_csv(requested) if metric in available]
    if not selected:
        raise ValueError(f"No requested metrics are available. Available metrics: {', '.join(available)}")
    return selected


def _plot_overview(
    rows: list[dict[str, Any]],
    metrics: list[str],
    prefixes: list[str],
    out_path: Path,
    *,
    title: str,
    smooth_window: int,
    fmt: str,
    dpi: int,
) -> None:
    plt = _setup_matplotlib()
    n = len(metrics)
    cols = 2 if n > 1 else 1
    rows_n = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.2 * cols, 4.2 * rows_n), squeeze=False)
    for ax, metric in zip(axes.flatten(), metrics):
        _plot_metric(ax, rows, metric, prefixes, smooth_window=smooth_window)
    for ax in axes.flatten()[n:]:
        ax.axis("off")
    fig.suptitle(title or "TrackSlotNet training history", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path.with_suffix(f".{fmt}"), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_separate(
    rows: list[dict[str, Any]],
    metrics: Iterable[str],
    prefixes: list[str],
    out_dir: Path,
    *,
    smooth_window: int,
    fmt: str,
    dpi: int,
) -> None:
    plt = _setup_matplotlib()
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        _plot_metric(ax, rows, metric, prefixes, smooth_window=smooth_window)
        fig.tight_layout()
        fig.savefig(out_dir / f"history_{metric}.{fmt}", dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)


def _latest_values(rows: list[dict[str, Any]], metrics: list[str], prefixes: list[str]) -> dict[str, float]:
    latest = rows[-1]
    out: dict[str, float] = {}
    for prefix in prefixes:
        for metric in metrics:
            key = f"{prefix}_{metric}"
            if _is_number(latest.get(key)):
                out[key] = float(latest[key])
    return out


def _best_values(rows: list[dict[str, Any]], metrics: list[str], prefixes: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    higher_is_better = {"track_f1", "track_precision", "track_recall", "count_acc", "max_objectness"}
    for prefix in prefixes:
        for metric in metrics:
            key = f"{prefix}_{metric}"
            values = [(float(row["epoch"]), float(row[key])) for row in rows if _is_number(row.get(key))]
            if not values:
                continue
            if metric in higher_is_better:
                epoch, value = max(values, key=lambda item: item[1])
            else:
                epoch, value = min(values, key=lambda item: item[1])
            out[key] = {"epoch": epoch, "value": value}
    return out


def main() -> int:
    args = parse_args()
    history_path = _resolve_history_path(args)
    rows = _read_history(history_path)
    prefixes = _split_csv(args.prefixes) or ["train", "val"]
    available = _available_base_metrics(rows, prefixes)
    if bool(args.list_metrics):
        print("\n".join(available))
        return 0

    metrics = _select_metrics(rows, prefixes, str(args.metrics))
    out_dir = _resolve_out_dir(args, history_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = str(args.format)
    overview_path = out_dir / "history_overview"
    _plot_overview(
        rows,
        metrics,
        prefixes,
        overview_path,
        title=str(args.title),
        smooth_window=int(args.smooth_window),
        fmt=fmt,
        dpi=int(args.dpi),
    )
    if bool(args.separate):
        _plot_separate(
            rows,
            metrics,
            prefixes,
            out_dir,
            smooth_window=int(args.smooth_window),
            fmt=fmt,
            dpi=int(args.dpi),
        )

    summary = {
        "history": str(history_path),
        "out_dir": str(out_dir),
        "epochs": [float(row["epoch"]) for row in rows if _is_number(row.get("epoch"))],
        "available_metrics": available,
        "selected_metrics": metrics,
        "prefixes": prefixes,
        "latest": _latest_values(rows, metrics, prefixes),
        "best": _best_values(rows, metrics, prefixes),
        "overview": str(overview_path.with_suffix(f".{fmt}")),
    }
    (out_dir / "history_plot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: epochs={len(rows)}, metrics={','.join(metrics)}, out={overview_path.with_suffix(f'.{fmt}')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
