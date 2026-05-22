"""Kalman-filter trajectory tracking adapted from `KF03.py` for real `.npy` DAS data.

用途：
    将 `/Volumes/SanDisk2T4/MyProjects/BaFang/KF/KF03.py` 中的核心卡尔曼滤波
    轨迹跟踪逻辑整理成一个独立、可复用、可命令行运行的模块，并添加直接读取
    真实 DAS `.npy` 数据的能力。该脚本不依赖 SAC 输入，适合直接在
    `/Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy` 这样的真实数组上
    进行单条种子轨迹跟踪。

用法示例：
    1. 直接对默认真实数据做一次从某个种子点出发的卡尔曼跟踪：

    ```bash
    uvr -m autotrack.kalman.kf03_real_npy \
      --seed-channel-idx 25 \
      --seed-time-index 180000 \
      --out-dir /tmp/kf03_real_npy_demo
    ```

    2. 显式指定输入 `.npy`、数组布局和输出目录：

    ```bash
    uvr -m autotrack.kalman.kf03_real_npy \
      --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy \
      --array-layout time_channel \
      --fs-hz 1000 \
      --dx-m 100 \
      --seed-channel-idx 30 \
      --seed-time-seconds 245.5 \
      --end-channel-idx 49 \
      --plot \
      --out-dir /tmp/kf03_real_npy_track
    ```

命令行参数说明：
    --input:
        真实 DAS `.npy` 文件路径。默认即 `00gauss_large.npy`。
    --array-layout:
        输入数组布局，`time_channel` 表示 `[time, channel]`，
        `channel_time` 表示 `[channel, time]`。
    --fs-hz:
        采样率（Hz）。
    --dx-m:
        通道间距（m）。
    --channel-start / --channel-count:
        读取时保留的通道范围。
    --time-start / --time-count:
        读取时保留的时间范围。
    --seed-channel-idx:
        种子轨迹所在的起始通道索引（相对于读取后的局部通道索引）。
    --seed-time-index / --seed-time-seconds:
        种子轨迹的时间位置。二者至少提供一个。
    --end-channel-idx:
        跟踪终止通道索引（相对于读取后的局部通道索引）。
    --sigma-a:
        状态过程噪声强度。
    --plot:
        是否输出一张带轨迹叠加的 PNG 图。
    --out-dir:
        输出目录。

输出结果：
    - `summary.json`：输入信息、种子信息、有效轨迹点数量等摘要。
    - `tracked_states.npy`：主轨迹点数组，shape `[1, n_channel]`。
    - `weak_states.npy`：弱观测 / 预测轨迹，shape `[1, n_channel]`。
    - `track_overlay.png`：可选，轨迹叠加图，字体使用 Times New Roman。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


DEFAULT_REAL_NPY = "/Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy"


def interpolate_middle_nans(veh_state: np.ndarray) -> np.ndarray:
    """Interpolate only middle NaNs while keeping leading/trailing NaNs."""
    veh_state = np.asarray(veh_state, dtype=float)
    isnan = np.isnan(veh_state)
    if not np.any(~isnan):
        return veh_state.copy()

    first_valid = int(np.argmax(~isnan))
    last_valid = int(len(veh_state) - np.argmax(~isnan[::-1]) - 1)
    interp_vals = veh_state.copy()
    x = np.arange(len(veh_state))
    valid_mask = ~isnan[first_valid : last_valid + 1]
    x_valid = x[first_valid : last_valid + 1][valid_mask]
    y_valid = veh_state[first_valid : last_valid + 1][valid_mask]
    interp_vals[first_valid : last_valid + 1] = np.interp(
        x[first_valid : last_valid + 1],
        x_valid,
        y_valid,
    )
    return interp_vals


def fit_and_fill_nans(arr: np.ndarray, deg: int = 2) -> np.ndarray:
    """Fill NaNs by polynomial fitting using the non-NaN entries."""
    arr = np.asarray(arr, dtype=float)
    x_all = np.arange(len(arr))
    mask = ~np.isnan(arr)
    if int(mask.sum()) < int(deg + 1):
        return arr

    coeffs = np.polyfit(x_all[mask], arr[mask], deg)
    poly = np.poly1d(coeffs)
    filled = arr.copy()
    filled[~mask] = poly(x_all[~mask])
    filled[filled < 0] = np.nan
    return filled


def build_default_args() -> dict:
    """Build default detector / vehicle arguments close to the original script."""
    return {
        "detect": {
            "prominence": 0.4,
            "distance": 500,
            "wlen": 1000,
            "height": None,
            "center_mode": "energy_center",
        },
        "veh": {
            "vel_init": 25.0,
            "vmin": 20.0,
            "vmax": 30.0,
            "tmin": -4.0,
            "tmax": 6.0,
            "dt": 0.001,
        },
    }


def load_real_npy(
    input_path: str | Path,
    *,
    array_layout: str = "time_channel",
    fs_hz: float = 1000.0,
    dx_m: float = 100.0,
    channel_start: int = 0,
    channel_count: int | None = 50,
    time_start: int = 0,
    time_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a real DAS `.npy` array and return `[channel, time]`, `t_axis`, `x_axis`."""
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Input `.npy` not found: {path}")

    arr = np.load(str(path), mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D `.npy` array, got shape {arr.shape}")

    if array_layout == "time_channel":
        time_dim, channel_dim = int(arr.shape[0]), int(arr.shape[1])
        ch0 = int(max(0, channel_start))
        ch1 = int(channel_dim if channel_count is None else min(channel_dim, ch0 + int(channel_count)))
        t0 = int(max(0, time_start))
        t1 = int(time_dim if time_count is None else min(time_dim, t0 + int(time_count)))
        data = np.array(arr[t0:t1, ch0:ch1], dtype=np.float32, copy=True).T
    elif array_layout == "channel_time":
        channel_dim, time_dim = int(arr.shape[0]), int(arr.shape[1])
        ch0 = int(max(0, channel_start))
        ch1 = int(channel_dim if channel_count is None else min(channel_dim, ch0 + int(channel_count)))
        t0 = int(max(0, time_start))
        t1 = int(time_dim if time_count is None else min(time_dim, t0 + int(time_count)))
        data = np.array(arr[ch0:ch1, t0:t1], dtype=np.float32, copy=True)
    else:
        raise ValueError("array_layout must be `time_channel` or `channel_time`")

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    t_axis = (np.arange(data.shape[1], dtype=np.float64) + float(time_start)) / float(fs_hz)
    x_axis = (np.arange(data.shape[0], dtype=np.float64) + float(channel_start)) * float(dx_m)
    return data, t_axis, x_axis


class KFTracking:
    """Space-domain Kalman tracking adapted from the original `KF03.py`."""

    def __init__(self, data: np.ndarray, t_axis: np.ndarray, x_axis: np.ndarray, args: dict):
        self.data = np.asarray(data, dtype=np.float32)
        self.t_axis = np.asarray(t_axis, dtype=np.float64)
        self.x_axis = np.asarray(x_axis, dtype=np.float64)
        self.dx = float(self.x_axis[1] - self.x_axis[0]) if len(self.x_axis) >= 2 else 100.0
        self.args = dict(args)

    def detect_in_one_section(self, start_x: float, nx: int = 1, pick_args: dict | None = None) -> np.ndarray:
        """Detect initial vehicle seeds around one spatial section."""
        if pick_args is None:
            pick_args = self.args["detect"]
        prominence = pick_args["prominence"]
        distance = pick_args["distance"]
        wlen = pick_args["wlen"]
        height = pick_args.get("height", None)

        start_x_idx = int(np.argmin(np.abs(start_x - self.x_axis)))
        all_peaks: list[int] = []
        for idx in range(int(nx)):
            row_idx = int(np.clip(start_x_idx + idx, 0, self.data.shape[0] - 1))
            peaks = find_peaks(
                self.data[row_idx],
                prominence=prominence,
                wlen=wlen,
                height=height,
                distance=distance,
            )[0]
            all_peaks.extend(int(p) for p in peaks)

        if not all_peaks:
            return np.array([], dtype=np.int32)

        all_peaks = np.array(sorted(all_peaks), dtype=np.int32)
        selected_peaks = [int(all_peaks[0])]
        min_interval = 3000
        for peak in all_peaks[1:]:
            if np.min(np.abs(np.asarray(selected_peaks, dtype=np.int32) - int(peak))) > min_interval:
                selected_peaks.append(int(peak))
        return np.asarray(selected_peaks, dtype=np.int32)

    def _init_state(self, veh_states: np.ndarray, start_x_idx: int, veh_base: np.ndarray):
        n_veh = len(veh_base)
        tkk = np.full((2, n_veh), np.nan, dtype=np.float64)
        pkk = np.full((2, 2, n_veh), np.nan, dtype=np.float64)
        xv = np.full(n_veh, np.nan, dtype=np.float64)
        for veh_idx in range(n_veh):
            val = veh_states[veh_idx, ~np.isnan(veh_states[veh_idx])]
            if len(val) > 0:
                tkk[:, veh_idx] = [val[0], 25.0]
                pkk[:, :, veh_idx] = np.array([[100.0, 0.0], [0.0, 4.0]], dtype=np.float64)
                xv[veh_idx] = self.x_axis[start_x_idx]
        return tkk, pkk, xv

    @staticmethod
    def _predict_state(tkk_v: np.ndarray, pkk_v: np.ndarray, dx: float, sigma_a: float):
        a = np.array([[1.0, dx], [0.0, 1.0]], dtype=np.float64)
        q = float(sigma_a) * np.array(
            [[0.25 * dx**4, 0.5 * dx**3], [0.5 * dx**3, 0.1 * dx**2]],
            dtype=np.float64,
        )
        tk1k = a @ tkk_v
        pk1k = a @ pkk_v @ a.T + q
        return tk1k, pk1k

    @staticmethod
    def _select_center(
        data_row: np.ndarray,
        pred_pos: float,
        tmin: int = -4000,
        tmax: int = 6000,
        mode: str = "energy_center",
    ) -> int | None:
        n = len(data_row)
        left = max(0, int(np.floor(pred_pos + tmin)))
        right = min(n - 1, int(np.ceil(pred_pos + tmax)))
        if left >= right:
            return None
        seg = data_row[left : right + 1]
        if mode == "main_peak":
            return int(left + np.argmax(np.abs(seg)))
        weights = np.abs(seg) ** 2
        wsum = float(np.sum(weights))
        if wsum == 0:
            return None
        center = float(np.sum(weights * np.arange(left, right + 1)) / wsum)
        return int(np.round(center))

    @staticmethod
    def _update_state(
        tk1k: np.ndarray,
        pk1k: np.ndarray,
        obs: float,
        c: np.ndarray,
        r: float,
        vmin: float = 20.0,
        vmax: float = 30.0,
    ):
        k = pk1k @ c / (float(r) + c @ pk1k @ c)
        tkk = tk1k + k * (float(obs) - c @ tk1k)
        tkk[1] = np.clip(tkk[1], float(vmin), float(vmax))
        pkk = pk1k - np.outer(k, c) @ pk1k
        return tkk, pkk

    def tracking_with_veh_base(
        self,
        start_x: float,
        end_x: float,
        veh_base: np.ndarray,
        sigma_a: float = 0.01,
        pick_args: dict | None = None,
        veh_args: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track one or more vehicles from seed time indices in space domain."""
        if pick_args is None:
            pick_args = self.args["detect"]
        if veh_args is None:
            veh_args = self.args["veh"]

        start_x_idx = int(np.argmin(np.abs(start_x - self.x_axis)))
        end_x_idx = int(np.argmin(np.abs(end_x - self.x_axis)))
        if end_x_idx < start_x_idx:
            start_x_idx, end_x_idx = end_x_idx, start_x_idx

        n_veh = len(veh_base)
        n_stat = int(end_x_idx - start_x_idx + 1)
        veh_weak_states = np.full((n_veh, n_stat), np.nan, dtype=np.float64)
        veh_states = np.full((n_veh, n_stat), np.nan, dtype=np.float64)
        veh_base_state = np.asarray(veh_base, dtype=np.float64).copy()
        nan_streak_count = np.zeros(n_veh, dtype=np.int32)
        terminate_flag = np.zeros(n_veh, dtype=bool)
        tkk, pkk, xv = self._init_state(veh_states, start_x_idx, veh_base)

        center_mode = str(pick_args.get("center_mode", "energy_center"))
        vel_init = float(veh_args.get("vel_init", 25.0))
        vmin = float(veh_args.get("vmin", 20.0))
        vmax = float(veh_args.get("vmax", 30.0))
        tmin = int(round(float(veh_args.get("tmin", -4.0)) / float(veh_args.get("dt", 0.001))))
        tmax = int(round(float(veh_args.get("tmax", 6.0)) / float(veh_args.get("dt", 0.001))))

        c = np.array([1.0, 0.0], dtype=np.float64)
        r = 10.0
        tkk[1, :] = vel_init

        for x_idx in range(start_x_idx, end_x_idx + 1):
            xi = float(self.x_axis[x_idx])
            for veh_idx in range(n_veh):
                if terminate_flag[veh_idx]:
                    continue

                if x_idx == start_x_idx:
                    tkk[:, veh_idx] = [float(veh_base[veh_idx]), vel_init]
                    pkk[:, :, veh_idx] = np.array([[625.0, 0.0], [0.0, 25.0]], dtype=np.float64)
                    xv[veh_idx] = xi
                    veh_base_state[veh_idx] = float(veh_base[veh_idx])
                    veh_states[veh_idx, 0] = float(veh_base[veh_idx])

                valid = veh_states[veh_idx, ~np.isnan(veh_states[veh_idx])]
                if len(valid) == 1:
                    tkk[:, veh_idx] = [float(valid[0]), vel_init]
                    pkk[:, :, veh_idx] = np.array([[625.0, 0.0], [0.0, 25.0]], dtype=np.float64)
                    xv[veh_idx] = float(self.x_axis[start_x_idx])
                    dx = xi - xv[veh_idx]
                    tk1k_v, pk1k_v = self._predict_state(tkk[:, veh_idx], pkk[:, :, veh_idx], dx, sigma_a)
                    veh_base_state[veh_idx] = tk1k_v[0]
                    tkk[:, veh_idx], pkk[:, :, veh_idx] = tk1k_v, pk1k_v
                elif len(valid) > 1:
                    dx = xi - xv[veh_idx]
                    tk1k_v, pk1k_v = self._predict_state(tkk[:, veh_idx], pkk[:, :, veh_idx], dx, sigma_a)
                    veh_base_state[veh_idx] = tk1k_v[0]
                    tkk[:, veh_idx], pkk[:, :, veh_idx] = tk1k_v, pk1k_v

                pred_pos = float(veh_base_state[veh_idx])
                center_idx = self._select_center(self.data[x_idx], pred_pos, tmin, tmax, mode=center_mode)
                rel_idx = int(x_idx - start_x_idx)
                if center_idx is not None:
                    veh_states[veh_idx, rel_idx] = float(center_idx)
                    nan_streak_count[veh_idx] = 0
                else:
                    veh_states[veh_idx, rel_idx] = np.nan
                    nan_streak_count[veh_idx] += 1
                    if int(nan_streak_count[veh_idx]) >= 4:
                        terminate_flag[veh_idx] = True

                veh_weak_states[veh_idx, rel_idx] = pred_pos
                if (not np.isnan(veh_states[veh_idx, rel_idx])) and int(np.sum(~np.isnan(veh_states[veh_idx]))) > 2:
                    obs = float(veh_states[veh_idx, rel_idx])
                    tkk[:, veh_idx], pkk[:, :, veh_idx] = self._update_state(
                        tkk[:, veh_idx],
                        pkk[:, :, veh_idx],
                        obs,
                        c,
                        r,
                        vmin,
                        vmax,
                    )
                    xv[veh_idx] = xi

        return veh_states, veh_weak_states


def plot_track_overlay(
    data: np.ndarray,
    t_axis: np.ndarray,
    x_axis: np.ndarray,
    tracked_states: np.ndarray,
    out_png: str | Path,
) -> str:
    """Render a simple wiggle/point overlay plot for one tracked state array."""
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(14, 7))
    offsets_km = np.asarray(x_axis, dtype=np.float64) * 1e-3
    spacing = float(np.median(np.diff(offsets_km))) if len(offsets_km) >= 2 else 0.1
    spacing = spacing if np.isfinite(spacing) and spacing > 0 else 0.1
    wiggle_amp = 0.25 * spacing
    data = np.asarray(data, dtype=np.float64)
    ref = float(np.quantile(np.abs(data[np.isfinite(data)]), 0.995)) if np.isfinite(data).any() else 1.0
    ref = max(ref, 1e-12)

    for ch_idx in range(data.shape[0]):
        ratio = np.clip(data[ch_idx] / ref, -1.25, 1.25)
        ax.plot(t_axis, offsets_km[ch_idx] + ratio * wiggle_amp, color="0.55", linewidth=0.65, alpha=0.8)

    if tracked_states.ndim == 2:
        track = tracked_states[0]
    else:
        track = tracked_states
    cols = np.where(~np.isnan(track))[0]
    if len(cols) > 0:
        t_idx = np.asarray(track[cols], dtype=np.int64)
        t_idx = np.clip(t_idx, 0, len(t_axis) - 1)
        ax.scatter(t_axis[t_idx], offsets_km[cols], color="#d62728", marker="x", s=36, linewidths=1.0)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Offset [km]")
    ax.set_title("Kalman Tracking Overlay")
    ax.grid(alpha=0.18)
    out_path = Path(out_png).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KF03-style Kalman tracking directly on real DAS `.npy` data.")
    parser.add_argument("--input", type=str, default=DEFAULT_REAL_NPY, help="Real DAS `.npy` input path.")
    parser.add_argument("--array-layout", choices=["time_channel", "channel_time"], default="time_channel", help="Input array layout.")
    parser.add_argument("--fs-hz", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel to load.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of channels to load.")
    parser.add_argument("--time-start", type=int, default=0, help="First time sample to load.")
    parser.add_argument("--time-count", type=int, default=240000, help="Number of time samples to load.")
    parser.add_argument("--seed-channel-idx", type=int, required=True, help="Seed channel index in the loaded local window.")
    parser.add_argument("--seed-time-index", type=int, default=None, help="Seed time index in the loaded local window.")
    parser.add_argument("--seed-time-seconds", type=float, default=None, help="Seed time in seconds in the loaded local window.")
    parser.add_argument("--end-channel-idx", type=int, default=None, help="End channel index in the loaded local window.")
    parser.add_argument("--sigma-a", type=float, default=0.0001, help="Process noise scale used by the Kalman predictor.")
    parser.add_argument("--fit-fill-nans", action="store_true", help="Also export a polynomial-filled version of the main track.")
    parser.add_argument("--plot", action="store_true", help="Save a PNG overlay plot.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_time_index = args.seed_time_index
    if seed_time_index is None:
        if args.seed_time_seconds is None:
            raise ValueError("Please provide either --seed-time-index or --seed-time-seconds")
        seed_time_index = int(round(float(args.seed_time_seconds) * float(args.fs_hz)))

    data, t_axis, x_axis = load_real_npy(
        args.input,
        array_layout=args.array_layout,
        fs_hz=float(args.fs_hz),
        dx_m=float(args.dx_m),
        channel_start=int(args.channel_start),
        channel_count=int(args.channel_count) if args.channel_count is not None else None,
        time_start=int(args.time_start),
        time_count=int(args.time_count) if args.time_count is not None else None,
    )
    if data.size == 0:
        raise ValueError("Loaded data is empty")

    seed_channel_idx = int(np.clip(args.seed_channel_idx, 0, data.shape[0] - 1))
    end_channel_idx = int(data.shape[0] - 1 if args.end_channel_idx is None else np.clip(args.end_channel_idx, 0, data.shape[0] - 1))

    tracker = KFTracking(data=data, t_axis=t_axis, x_axis=x_axis, args=build_default_args())
    tracked_states, weak_states = tracker.tracking_with_veh_base(
        start_x=float(x_axis[seed_channel_idx]),
        end_x=float(x_axis[end_channel_idx]),
        veh_base=np.array([int(seed_time_index)], dtype=np.float64),
        sigma_a=float(args.sigma_a),
    )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tracked_path = out_dir / "tracked_states.npy"
    weak_path = out_dir / "weak_states.npy"
    np.save(tracked_path, tracked_states)
    np.save(weak_path, weak_states)

    fitted_path = None
    if args.fit_fill_nans and tracked_states.size > 0:
        filled = fit_and_fill_nans(interpolate_middle_nans(tracked_states[0]))
        fitted_path = out_dir / "tracked_states_filled.npy"
        np.save(fitted_path, filled)

    plot_path = None
    if args.plot:
        plot_path = plot_track_overlay(
            data=data,
            t_axis=t_axis,
            x_axis=x_axis,
            tracked_states=tracked_states,
            out_png=out_dir / "track_overlay.png",
        )

    valid_points = int(np.sum(~np.isnan(tracked_states[0]))) if tracked_states.size > 0 else 0
    summary = {
        "input": str(Path(args.input).expanduser()),
        "array_layout": str(args.array_layout),
        "fs_hz": float(args.fs_hz),
        "dx_m": float(args.dx_m),
        "channel_start": int(args.channel_start),
        "channel_count": int(data.shape[0]),
        "time_start": int(args.time_start),
        "time_count": int(data.shape[1]),
        "seed_channel_idx": int(seed_channel_idx),
        "seed_time_index": int(seed_time_index),
        "end_channel_idx": int(end_channel_idx),
        "sigma_a": float(args.sigma_a),
        "valid_points": int(valid_points),
        "tracked_states_path": str(tracked_path),
        "weak_states_path": str(weak_path),
        "filled_states_path": str(fitted_path) if fitted_path is not None else "",
        "plot_path": str(plot_path) if plot_path is not None else "",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
