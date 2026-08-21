from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot vehicle peak-set training loss curves.")
    parser.add_argument("--history", required=True, type=Path, help="Path to train_history.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="Output PNG path")
    return parser.parse_args(argv)


def _load_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"empty history: {path}")
    return rows


def _series(rows: list[dict[str, Any]], split: str, key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        part = row.get(split, {})
        if key not in part:
            continue
        xs.append(int(row["epoch"]))
        ys.append(float(part[key]))
    return xs, ys


def _plot_metric_group(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    curves: list[tuple[str, str, str]],
    *,
    title: str,
) -> None:
    styles = {
        "o": {"marker": "o", "linestyle": "-", "linewidth": 1.45},
        "s": {"marker": "s", "linestyle": "-", "linewidth": 1.45},
        "D": {"marker": "D", "linestyle": "-", "linewidth": 1.45},
        "*": {"marker": "*", "linestyle": "-", "linewidth": 1.35},
        "^": {"marker": "^", "linestyle": "-", "linewidth": 1.35},
        "v": {"marker": "v", "linestyle": "-", "linewidth": 1.35},
        "P": {"marker": "P", "linestyle": "-", "linewidth": 1.35},
        "X": {"marker": "X", "linestyle": "-", "linewidth": 1.35},
    }
    for key, label, marker in curves:
        tx, ty = _series(rows, "train", key)
        vx, vy = _series(rows, "val", key)
        style = dict(styles.get(marker, {"marker": marker, "linestyle": "-", "linewidth": 1.35}))
        if tx:
            ax.plot(tx, ty, label=f"train {label}", **style)
        if vx:
            val_style = dict(style)
            val_style["linestyle"] = "--"
            ax.plot(vx, vy, label=f"val {label}", **val_style)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncols=2, fontsize=8)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _load_history(args.history)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train"]["loss"]) for row in rows if "train" in row and "loss" in row["train"]]
    val_loss = [float(row["val"]["loss"]) for row in rows if "val" in row and "loss" in row["val"]]

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 13.0), dpi=170, constrained_layout=True)

    ax = axes[0]
    ax.plot(epochs[: len(train_loss)], train_loss, marker="o", linewidth=1.8, label="train loss")
    ax.plot(epochs[: len(val_loss)], val_loss, marker="s", linewidth=1.8, label="val loss")
    ax.set_title("Vehicle peak-set training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    _plot_metric_group(
        axes[1],
        rows,
        [
            ("loss_obj", "obj", "o"),
            ("loss_anchor", "anchor", "D"),
            ("loss_anchor_time", "anchor_time", "s"),
            ("loss_complete_valid", "complete_valid", "*"),
            ("loss_observed_valid", "observed_valid", "P"),
        ],
        title="Recognition losses",
    )
    _plot_metric_group(
        axes[2],
        rows,
        [
            ("loss_complete_time", "complete_time", "o"),
            ("loss_missing_complete_time", "missing_time", "s"),
            ("loss_missing_complete_valid", "missing_valid", "*"),
            ("loss_dead_complete_time", "dead_time", "D"),
            ("loss_dead_complete_valid", "dead_valid", "^"),
        ],
        title="Completion losses",
    )
    _plot_metric_group(
        axes[3],
        rows,
        [
            ("loss_inertia", "inertia", "s"),
            ("loss_jump", "jump", "X"),
            ("loss_smooth", "smooth", "D"),
            ("loss_slope_variation", "slope_variation", "P"),
            ("loss_monotonic", "monotonic", "^"),
        ],
        title="Shape losses",
    )

    fig.savefig(str(out), bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"history": str(args.history), "out": str(out), "epochs": len(rows)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
