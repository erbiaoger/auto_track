"""Kalman tracking GUI adapted from the original `KF03.py` for real `.npy` data.

用途：
    将原始 `KF03.py` 的交互式卡尔曼滤波标注流程完整迁移到当前项目中，
    并直接支持读取真实 DAS `.npy` 数据。该 GUI 面向“以卡尔曼滤波为主”的
    交互标注，不依赖当前图搜索 GUI 的点击逻辑。

主要交互：
    - 左键：从当前点击点的最近峰值出发，执行一次卡尔曼滤波轨迹跟踪。
    - 中键：将当前临时轨迹在该通道上吸附到最近峰值。
    - 右键：两次点击之间补全轨迹；也支持裁剪模式。
    - 滚轮：平移窗口；`Shift` / `Cmd` / `Ctrl` + 滚轮：缩放窗口。
    - `S`：将当前临时轨迹加入当前窗口的轨迹列表。
    - `F`：保存当前窗口的数据和轨迹，并向后移动一个窗口步长。
    - `Backspace`：删除最后一条已保存轨迹。
    - `X`：清空当前临时轨迹。
    - `Z`：对当前临时轨迹做多项式补 NaN。
    - `C`：切换质心模式 `energy_center / main_peak`。
    - `E`：右键进入裁剪模式。
    - `R`：右键恢复补全模式。

用例：
    uvr -m autotrack.gui.kalman_track_gui \
      --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy \
      --out-dir /tmp/kf_gui_labels

输出：
    每次按 `F` 时，会在输出目录写一个窗口 bundle：
    - `data.npy`
    - `tracks.npy`
    - `meta.json`
    - `overlay.png`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy.signal import find_peaks

from autotrack.kalman import KFTracking, build_default_args, fit_and_fill_nans, interpolate_middle_nans, load_real_npy


plt.rcParams["font.family"] = "Times New Roman"


class KalmanTrackGUI(QMainWindow):
    def __init__(
        self,
        *,
        input_path: str,
        out_dir: str,
        array_layout: str = "time_channel",
        fs_hz: float = 1000.0,
        dx_m: float = 100.0,
        channel_start: int = 0,
        channel_count: int = 50,
        time_start: int = 0,
        time_count: int | None = None,
        window_seconds: float = 240.0,
        step_seconds: float = 120.0,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Kalman Vehicle Tracking GUI")

        self.input_path = str(Path(input_path).expanduser())
        self.out_dir = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.array_layout = str(array_layout)
        self.fs_hz = float(fs_hz)
        self.dx_m = float(dx_m)
        self.channel_start = int(channel_start)
        self.channel_count = int(channel_count)
        self.time_start = int(time_start)
        self.time_count = None if time_count is None else int(time_count)

        self.data_all, self.full_t_axis, self.full_x_axis = load_real_npy(
            self.input_path,
            array_layout=self.array_layout,
            fs_hz=self.fs_hz,
            dx_m=self.dx_m,
            channel_start=self.channel_start,
            channel_count=self.channel_count,
            time_start=self.time_start,
            time_count=self.time_count,
        )

        self.default_window_size = int(round(float(window_seconds) * float(self.fs_hz)))
        self.window_size = max(1000, min(self.default_window_size, int(self.data_all.shape[1])))
        self.current_start = 0
        self.scroll_step_s = float(step_seconds)
        self.center_mode = "energy_center"
        self.right_click_mode = "fill"  # fill | cut
        self.window_save_index = 0

        self.fig = Figure(figsize=(12, 7))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.update_tracking_data(reset_states=True)

    def _plot_section(self) -> None:
        data_plot = np.asarray(self.tracking.data, dtype=np.float64)
        n_ch = int(data_plot.shape[0])
        t = np.asarray(self.tracking.t_axis, dtype=np.float64)
        offsets_km = np.asarray(self.tracking.x_axis, dtype=np.float64) * 1e-3
        if offsets_km.size >= 2:
            spacing = float(np.median(np.diff(offsets_km)))
            if not np.isfinite(spacing) or spacing <= 0:
                spacing = 0.1
        else:
            spacing = 0.1
        wiggle_amp = 0.27 * spacing
        eps = 1e-12

        abs_vals = np.abs(data_plot[np.isfinite(data_plot)])
        if abs_vals.size == 0:
            global_ref = 0.0
        else:
            q995 = float(np.quantile(abs_vals, 0.995))
            rms = float(np.sqrt(np.mean(abs_vals * abs_vals)))
            global_ref = max(q995, 3.0 * rms, eps)
        clip_ratio = 1.35

        for i in range(n_ch):
            trace = data_plot[i]
            if (not np.isfinite(global_ref)) or global_ref < eps:
                x = np.full_like(t, offsets_km[i], dtype=np.float64)
            else:
                ratio = np.clip(trace / global_ref, -clip_ratio, clip_ratio)
                x = offsets_km[i] + ratio * wiggle_amp
            self.ax.plot(x, t, color="0.45", linewidth=0.8, alpha=0.9)

    def redraw(self) -> None:
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self._plot_section()
        self.ax.set_ylim(self.tracking.t_axis[0], self.tracking.t_axis[-1])
        self.ax.invert_yaxis()
        y0 = self.tracking.x_axis[0] * 1e-3
        y1 = self.tracking.x_axis[-1] * 1e-3
        if self.tracking.x_axis.size >= 2:
            dy = float(np.median(np.diff(self.tracking.x_axis)) * 1e-3)
            if not np.isfinite(dy) or dy <= 0:
                dy = 0.1
        else:
            dy = 0.1
        pad = 0.4 * dy
        self.ax.set_xlim(y0 - pad, y1 + pad)
        self.ax.margins(x=0, y=0)
        t_offset = getattr(self, "t_offset", 0.0)
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f"{y + t_offset:.1f}"))
        self.ax.set_xlabel("Offset [km]")
        self.ax.set_ylabel("Time [s]")
        self.ax.set_title(
            f"Window=[{t_offset:.1f}, {t_offset + self.window_size / self.fs_hz:.1f}] s  "
            f"center_mode={self.center_mode}  right_click_mode={self.right_click_mode}"
        )

        if not np.isnan(self.veh_state).all():
            cols = np.where(~np.isnan(self.veh_state))[0]
            t_idx_global = self.veh_state[cols].astype(int)
            t_idx_local = t_idx_global - self.current_start
            in_view = (t_idx_local >= 0) & (t_idx_local < len(self.tracking.t_axis))
            if np.any(in_view):
                cols_in = cols[in_view]
                t_idx_local = t_idx_local[in_view]
                ts = self.tracking.t_axis[t_idx_local]
                xs = self.tracking.x_axis[cols_in] * 1e-3
                self.ax.plot(xs, ts, color="k", linewidth=1.2, alpha=0.9)
                if hasattr(self, "veh_peak_mask") and self.veh_peak_mask is not None:
                    peak_cols = cols_in[np.asarray(self.veh_peak_mask[cols_in], dtype=bool)]
                    if peak_cols.size > 0:
                        peak_t_local = self.veh_state[peak_cols].astype(int) - self.current_start
                        self.ax.scatter(
                            self.tracking.x_axis[peak_cols] * 1e-3,
                            self.tracking.t_axis[peak_t_local],
                            marker="o",
                            color="red",
                            s=30,
                            zorder=9,
                        )

        if self.veh_states:
            for idx_track, veh_state in enumerate(self.veh_states):
                t_values_global = veh_state[~np.isnan(veh_state)].astype(int)
                x_indices = np.where(~np.isnan(veh_state))[0]
                t_values_local = t_values_global - self.current_start
                in_view = (t_values_local >= 0) & (t_values_local < len(self.tracking.t_axis))
                if np.any(in_view):
                    xs = self.tracking.x_axis[x_indices[in_view]] * 1e-3
                    ts = self.tracking.t_axis[t_values_local[in_view]]
                    self.ax.plot(xs, ts, color="#1f77b4", linewidth=1.1, alpha=0.85)
                    self.ax.scatter(
                        xs,
                        ts,
                        marker="x",
                        s=48,
                    )
                    if idx_track < len(self.veh_peak_masks):
                        peak_mask = np.asarray(self.veh_peak_masks[idx_track], dtype=bool)
                        peak_cols = x_indices[in_view & peak_mask[x_indices]]
                        if peak_cols.size > 0:
                            peak_t_local = veh_state[peak_cols].astype(int) - self.current_start
                            self.ax.scatter(
                                self.tracking.x_axis[peak_cols] * 1e-3,
                                self.tracking.t_axis[peak_t_local],
                                marker="o",
                                color="red",
                                s=24,
                                zorder=8,
                            )

        self.canvas.draw_idle()

    def adjust_window_size(self, scale: float) -> None:
        new_size = int(self.window_size * float(scale))
        new_size = max(1000, min(new_size, int(self.data_all.shape[1])))
        if new_size == self.window_size:
            return
        self.window_size = new_size
        self.current_start = min(self.current_start, max(0, self.data_all.shape[1] - self.window_size))
        self.update_tracking_data(reset_states=False)

    def adjust_window_size_at_cursor(self, scale: float, cursor_x: float | None) -> None:
        if cursor_x is None:
            self.adjust_window_size(scale)
            return
        old_size = int(self.window_size)
        new_size = int(self.window_size * float(scale))
        new_size = max(1000, min(new_size, int(self.data_all.shape[1])))
        if new_size == old_size:
            return
        cursor_global_s = float(cursor_x) + float(self.t_offset)
        old_window_s = float(old_size) * self.dt
        rel = 0.0 if old_window_s == 0 else float(cursor_x) / old_window_s
        new_start_s = cursor_global_s - rel * (float(new_size) * self.dt)
        new_start = int(round(new_start_s / self.dt))
        new_start = max(0, min(new_start, int(self.data_all.shape[1] - new_size)))
        self.window_size = int(new_size)
        self.current_start = int(new_start)
        self.update_tracking_data(reset_states=False)

    def _scroll_window(self, delta_samples: int) -> bool:
        new_start = int(self.current_start) + int(delta_samples)
        new_start = max(0, min(new_start, int(self.data_all.shape[1] - self.window_size)))
        if new_start == self.current_start:
            return False
        self.current_start = int(new_start)
        self.update_tracking_data(reset_states=False)
        return True

    def on_scroll(self, event) -> None:
        if event is None:
            return
        if hasattr(event, "inaxes") and event.inaxes is not self.ax:
            return

        button = str(getattr(event, "button", "") or "").lower()
        if button == "up":
            step = 1
        elif button == "down":
            step = -1
        else:
            raw_step = getattr(event, "step", 0)
            if raw_step == 0:
                return
            step = 1 if raw_step > 0 else -1

        modifiers = None
        if hasattr(event, "guiEvent") and event.guiEvent is not None:
            try:
                modifiers = event.guiEvent.modifiers()
            except Exception:  # noqa: BLE001
                modifiers = None
        use_shift = bool(modifiers and (modifiers & Qt.KeyboardModifier.ShiftModifier))
        use_cmd = bool(modifiers and (modifiers & (Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.ControlModifier)))

        if use_shift or use_cmd:
            scale = 0.8 if step > 0 else 1.25
            self.adjust_window_size_at_cursor(scale, event.xdata)
        else:
            self._scroll_window(int(self.scroll_step_samples) * (-step))

    def _channel_pick_configs(self, x_idx: int, *, relaxed: bool = False) -> list[dict[str, float | int | None]]:
        if int(x_idx) < 0 or int(x_idx) >= int(self.tracking.data.shape[0]):
            return []
        trace = np.asarray(self.tracking.data[int(x_idx)], dtype=np.float64)
        finite = trace[np.isfinite(trace)]
        if finite.size == 0:
            return []
        q90, q95, q99 = np.quantile(finite, [0.90, 0.95, 0.99])
        dyn = max(float(q99 - q90), 1e-4)
        base_prom = max(0.008, min(0.06, 0.35 * dyn))
        if relaxed:
            base_prom *= 0.65
        configs: list[dict[str, float | int | None]] = [
            {"prominence": base_prom, "distance": 80, "wlen": 801, "height": None},
            {"prominence": max(0.005, 0.6 * base_prom), "distance": 48, "wlen": 401, "height": None},
            {"prominence": max(0.003, 0.4 * base_prom), "distance": 24, "wlen": 201, "height": None},
        ]
        if float(q95) > 0.0:
            configs.append(
                {
                    "prominence": max(0.003, 0.35 * base_prom),
                    "distance": 24,
                    "wlen": 201,
                    "height": float(0.7 * q95),
                }
            )
        return configs

    def _find_nearest_peak(
        self,
        x_idx: int,
        clicked_t: float,
        pick: dict | None = None,
        *,
        relaxed: bool = False,
    ) -> int | None:
        if int(x_idx) < 0 or int(x_idx) >= int(self.tracking.data.shape[0]):
            return None
        configs: list[dict[str, float | int | None]] = []
        if pick:
            cfg = {
                "prominence": float(pick.get("prominence", 0.01) or 0.01),
                "distance": int(pick.get("distance", 80) or 80),
                "wlen": int(pick.get("wlen", 801) or 801),
                "height": pick.get("height", None),
            }
            configs.append(cfg)
        configs.extend(self._channel_pick_configs(x_idx, relaxed=relaxed))

        for cfg in configs:
            peaks = find_peaks(
                self.tracking.data[x_idx],
                prominence=float(cfg["prominence"]),
                distance=max(1, int(cfg["distance"])),
                wlen=max(3, int(cfg["wlen"])),
                height=cfg.get("height", None),
            )[0]
            if peaks.size == 0:
                continue
            peak_times = self.tracking.t_axis[peaks]
            best_idx = int(np.argmin(np.abs(peak_times - float(clicked_t))))
            return int(peaks[best_idx])
        return None

    def _find_local_peak_near_index(
        self,
        x_idx: int,
        target_t_idx: int,
        *,
        search_radius: int = 20,
        relaxed: bool = True,
        include_abs_fallback: bool = True,
    ) -> int | None:
        if int(x_idx) < 0 or int(x_idx) >= int(self.tracking.data.shape[0]):
            return None
        target_t_idx = int(target_t_idx)
        radii = []
        base = max(1, int(search_radius))
        for factor in (1, 3, 8, 20):
            radius = base * factor
            if radius not in radii:
                radii.append(radius)
        for radius in radii:
            ti_low = max(0, target_t_idx - int(radius))
            ti_high = min(len(self.tracking.t_axis), target_t_idx + int(radius) + 1)
            if ti_high - ti_low < 3:
                continue
            for cfg in self._channel_pick_configs(x_idx, relaxed=relaxed):
                peaks = find_peaks(
                    self.tracking.data[x_idx][ti_low:ti_high],
                    prominence=float(cfg["prominence"]),
                    distance=max(1, int(cfg["distance"])),
                    wlen=max(3, min(int(cfg["wlen"]), max(3, ti_high - ti_low))),
                    height=cfg.get("height", None),
                )[0]
                if peaks.size == 0:
                    continue
                return int(peaks[np.argmin(np.abs(peaks - (target_t_idx - ti_low)))]) + ti_low
            # Fallback: accept the nearest true local maximum even if it fails prominence.
            trace_win = np.asarray(self.tracking.data[x_idx][ti_low:ti_high], dtype=np.float64)
            if trace_win.size < 3:
                continue
            local_maxima: list[int] = []
            for idx_local in range(1, int(trace_win.size) - 1):
                left = float(trace_win[idx_local - 1])
                center = float(trace_win[idx_local])
                right = float(trace_win[idx_local + 1])
                if center >= left and center >= right and (center > left or center > right):
                    local_maxima.append(idx_local)
            if local_maxima:
                local_arr = np.asarray(local_maxima, dtype=np.int64)
                nearest = int(local_arr[np.argmin(np.abs(local_arr - (target_t_idx - ti_low)))])
                return nearest + ti_low
            # Fallback: smooth slightly, then look for a local maximum on the smoothed trace.
            if trace_win.size >= 5:
                kernel = np.ones(5, dtype=np.float64) / 5.0
                smooth = np.convolve(trace_win, kernel, mode="same")
                smooth_maxima: list[int] = []
                for idx_local in range(1, int(smooth.size) - 1):
                    left = float(smooth[idx_local - 1])
                    center = float(smooth[idx_local])
                    right = float(smooth[idx_local + 1])
                    if center >= left and center >= right and (center > left or center > right):
                        smooth_maxima.append(idx_local)
                if smooth_maxima:
                    smooth_arr = np.asarray(smooth_maxima, dtype=np.int64)
                    nearest = int(smooth_arr[np.argmin(np.abs(smooth_arr - (target_t_idx - ti_low)))])
                    return nearest + ti_low
            if include_abs_fallback:
                abs_win = np.abs(trace_win)
                if abs_win.size > 0 and np.any(np.isfinite(abs_win)):
                    return int(np.nanargmax(abs_win)) + ti_low
        return None

    def _track_peak_mask(self, veh_state: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(veh_state, dtype=bool)
        for ch_idx, t_idx in enumerate(np.asarray(veh_state, dtype=np.float64)):
            if not np.isfinite(t_idx):
                continue
            if ch_idx < 0 or ch_idx >= int(self.tracking.data.shape[0]):
                continue
            local_t_idx = int(round(float(t_idx) - float(self.current_start)))
            peak_idx = self._find_local_peak_near_index(
                ch_idx,
                local_t_idx,
                search_radius=20,
                relaxed=True,
                include_abs_fallback=False,
            )
            if peak_idx is not None:
                mask[ch_idx] = True
        return mask

    def _refresh_current_peak_mask(self) -> None:
        self.veh_peak_mask = self._track_peak_mask(self.veh_state)

    def on_click(self, event) -> None:
        if not getattr(event, "inaxes", None):
            return

        clicked_x = float(event.xdata) * 1e3
        clicked_t = float(event.ydata)
        t_approx = int(np.argmin(np.abs(self.tracking.t_axis - clicked_t)))
        x_idx = int(np.argmin(np.abs(self.tracking.x_axis - clicked_x)))
        pick = self.tracking.args["detect"]

        peak_t_idx = self._find_nearest_peak(x_idx, clicked_t, pick, relaxed=True)
        t_idx = int(peak_t_idx) if peak_t_idx is not None else int(t_approx)

        if event.button == 1:
            if peak_t_idx is None:
                print("Warning: no peak found at clicked location.")
                return
            self.temp_points.append((x_idx, int(t_idx)))
            print(f"Seed point: x_idx={x_idx}, t_idx={t_idx}")
            try:
                new_states, _ = self.tracking.tracking_with_veh_base(
                    start_x=float(self.tracking.x_axis[x_idx]),
                    end_x=float(self.tracking.x_axis[-1]),
                    veh_base=np.array([int(t_idx)], dtype=np.float64),
                    sigma_a=0.0001,
                )
                track_len = int(new_states.shape[1])
                new_states_1d = interpolate_middle_nans(new_states[0])
                new_states_global = new_states_1d.copy()
                valid = ~np.isnan(new_states_global)
                new_states_global[valid] = new_states_global[valid] + float(self.current_start)
                end_idx = min(len(self.veh_state), x_idx + track_len)
                self.veh_state[x_idx:end_idx] = new_states_global[: end_idx - x_idx]
                self._refresh_current_peak_mask()
            except Exception as exc:  # noqa: BLE001
                print(f"Tracking failed: {exc}")
            self.redraw()

        elif event.button == 2:
            peak_t_idx = self._find_nearest_peak(x_idx, clicked_t, None, relaxed=True)
            if peak_t_idx is None:
                print("Warning: no peak found for middle click.")
                return
            self.veh_state[x_idx] = float(peak_t_idx + self.current_start)
            self._refresh_current_peak_mask()
            print(f"Middle-click snapped peak: x_idx={x_idx}, t_idx={peak_t_idx + self.current_start}")
            self.redraw()

        elif event.button == 3:
            if self.right_click_mode == "cut":
                self.veh_state[x_idx:] = np.nan
                self._refresh_current_peak_mask()
                print(f"Cut veh_state after x_idx={x_idx}")
                self.redraw()
                return

            if not hasattr(self, "right_click_tmp"):
                self.right_click_tmp = []

            if len(self.right_click_tmp) == 1:
                if peak_t_idx is None:
                    print("Warning: no peak found for right-click endpoint.")
                    return
                self.right_click_tmp.append((x_idx, int(peak_t_idx)))
                print(f"Right-click point 2: x_idx={x_idx}, t_idx={int(peak_t_idx)}")
                x_start_idx, _t_start_hint = self.right_click_tmp[0]
                x_start_m = float(self.tracking.x_axis[int(x_start_idx)])
                dist = np.abs(self.tracking.x_axis - x_start_m)
                idx = int(np.nanargmin(dist + np.isnan(self.veh_state) * 1e10))
                if np.isnan(self.veh_state[idx]):
                    print("Warning: no valid veh_state point found.")
                    self.right_click_tmp = []
                    return
                t0_local = int(self.veh_state[idx]) - int(self.current_start)
                if t0_local < 0 or t0_local >= len(self.tracking.t_axis):
                    print("Warning: seed point is outside current window.")
                    self.right_click_tmp = []
                    return
                snapped_t0 = self._find_local_peak_near_index(idx, t0_local, search_radius=20, relaxed=True)
                if snapped_t0 is None:
                    print("Warning: start point cannot be snapped to a real peak.")
                    self.right_click_tmp = []
                    return
                self.right_click_tmp[0] = (idx, int(snapped_t0))
                print(f"Right-click start snapped to nearest veh_state point: x_idx={idx}, t_idx={t0_local}")

            elif len(self.right_click_tmp) == 0:
                print("Right-click: choose an existing track vicinity first, then a peak endpoint.")
                self.right_click_tmp.append((x_idx, int(t_idx)))
                print(f"Right-click point 1: x_idx={x_idx}")

            elif len(self.right_click_tmp) == 2:
                x0, t0 = self.right_click_tmp[0]
                x1, t1 = self.right_click_tmp[1]
                if x0 > x1:
                    x0, x1 = x1, x0
                    t0, t1 = t1, t0
                print(f"Fill trajectory interval x[{x0}:{x1}]")
                added = 0
                skipped = 0
                for xi in range(int(x0), int(x1) + 1):
                    interp_t = float(t0) + (float(t1 - t0) * float(xi - x0) / max(int(x1 - x0), 1))
                    local_peak_idx = self._find_local_peak_near_index(
                        xi,
                        int(round(interp_t)),
                        search_radius=20,
                        relaxed=True,
                        include_abs_fallback=False,
                    )
                    if local_peak_idx is not None:
                        self.veh_state[xi] = float(local_peak_idx + self.current_start)
                        added += 1
                    else:
                        self.veh_state[xi] = float(interp_t + self.current_start)
                        skipped += 1
                self.right_click_tmp = []
                self._refresh_current_peak_mask()
                print(f"Fill finished: added={added}, skipped={skipped}")
                self.redraw()

    def _save_overlay_png(self, bundle_dir: Path) -> None:
        out_path = bundle_dir / "overlay.png"
        self.fig.savefig(out_path, dpi=180)

    def save_current_window_bundle(self) -> None:
        if not self.veh_states:
            print("Warning: no saved trajectories in current window.")
            return
        bundle_dir = self.out_dir / f"window_{self.window_save_index:04d}_t{self.current_start:09d}"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        np.save(bundle_dir / "data.npy", np.asarray(self.tracking.data, dtype=np.float32))
        np.save(bundle_dir / "tracks.npy", np.asarray(self.veh_states, dtype=np.float64))
        meta = {
            "input_path": self.input_path,
            "array_layout": self.array_layout,
            "fs_hz": float(self.fs_hz),
            "dx_m": float(self.dx_m),
            "channel_start": int(self.channel_start),
            "channel_count": int(self.tracking.data.shape[0]),
            "time_start_sample": int(self.current_start + self.time_start),
            "window_size_samples": int(self.window_size),
            "window_size_seconds": float(self.window_size) / float(self.fs_hz),
            "track_count": int(len(self.veh_states)),
            "center_mode": str(self.center_mode),
            "right_click_mode": str(self.right_click_mode),
        }
        (bundle_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self._save_overlay_png(bundle_dir)
        print(f"Saved current window bundle to {bundle_dir}")
        self.window_save_index += 1

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_C:
            self.center_mode = "main_peak" if self.center_mode == "energy_center" else "energy_center"
            print(f"center_mode = {self.center_mode}")
            self.update_tracking_data(reset_states=False)
            return
        if key == Qt.Key.Key_Down:
            self._scroll_window(int(self.scroll_step_samples))
            return
        if key == Qt.Key.Key_Up:
            self._scroll_window(-int(self.scroll_step_samples))
            return
        if key == Qt.Key.Key_PageDown:
            self._scroll_window(int(self.window_size))
            return
        if key == Qt.Key.Key_PageUp:
            self._scroll_window(-int(self.window_size))
            return
        if key == Qt.Key.Key_E:
            self.right_click_mode = "cut"
            print("Right-click cut mode enabled")
            return
        if key == Qt.Key.Key_R:
            self.right_click_mode = "fill"
            print("Right-click fill mode enabled")
            return
        if key == Qt.Key.Key_X:
            self.veh_state = np.full((len(self.tracking.x_axis)), np.nan)
            self.veh_peak_mask = np.zeros((len(self.tracking.x_axis),), dtype=bool)
            self.temp_points = []
            self.redraw()
            return
        if key == Qt.Key.Key_S:
            if not np.isnan(self.veh_state).all():
                self.veh_states.append(self.veh_state.copy())
                self.veh_peak_masks.append(self.veh_peak_mask.copy())
                print(f"Saved current temporary track #{len(self.veh_states)}")
                self.veh_state = np.full((len(self.tracking.x_axis)), np.nan)
                self.veh_peak_mask = np.zeros((len(self.tracking.x_axis),), dtype=bool)
                self.temp_points = []
                self.redraw()
            else:
                print("Current veh_state is empty; nothing saved.")
            return
        if key == Qt.Key.Key_Z:
            self.veh_state = fit_and_fill_nans(self.veh_state, deg=2)
            self._refresh_current_peak_mask()
            self.redraw()
            return
        if key == Qt.Key.Key_Backspace:
            if self.veh_states:
                self.veh_states.pop()
                if self.veh_peak_masks:
                    self.veh_peak_masks.pop()
                print("Deleted last saved trajectory")
                self.redraw()
            else:
                print("No saved trajectory to delete")
            return
        if key == Qt.Key.Key_F:
            self.save_current_window_bundle()
            step = int(round(self.scroll_step_s * self.fs_hz))
            new_start = min(int(self.data_all.shape[1] - self.window_size), int(self.current_start + step))
            if new_start == self.current_start:
                print("Reached the end of the data.")
                return
            self.current_start = int(new_start)
            self.update_tracking_data(reset_states=True)
            return
        super().keyPressEvent(event)

    def update_tracking_data(self, reset_states: bool = True) -> None:
        end = int(self.current_start + self.window_size)
        end = min(end, int(self.data_all.shape[1]))
        data = self.data_all[:, self.current_start:end]
        x_axis = np.arange(len(data), dtype=np.float64) * float(self.dx_m)
        t_axis = np.arange(data.shape[1], dtype=np.float64) / float(self.fs_hz)
        self.t_offset = float(self.current_start) / float(self.fs_hz)
        self.dt = float(1.0 / self.fs_hz)
        self.scroll_step_samples = max(1, int(self.scroll_step_s / self.dt))

        args = build_default_args()
        args["detect"]["prominence"] = 0.01
        args["detect"]["distance"] = 80
        args["detect"]["wlen"] = 801
        args["detect"]["height"] = None
        args["detect"]["center_mode"] = self.center_mode
        args["veh"]["vmin"] = 22.0
        args["veh"]["vmax"] = 28.0
        args["veh"]["dx"] = float(self.dx_m)
        self.tracking = KFTracking(data=data, t_axis=t_axis, x_axis=x_axis, args=args)

        if reset_states:
            self.manual_states = []
            self.temp_points = []
            self.base_states = []
            self.veh_states: list[np.ndarray] = []
            self.veh_peak_masks: list[np.ndarray] = []
            self.veh_state = np.full((len(self.tracking.x_axis)), np.nan, dtype=np.float64)
            self.veh_peak_mask = np.zeros((len(self.tracking.x_axis),), dtype=bool)
            self.right_click_tmp = []

        self.redraw()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kalman tracking GUI for real DAS `.npy` data.")
    parser.add_argument("--input", type=str, default="/Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy", help="Real DAS `.npy` input.")
    parser.add_argument("--out-dir", type=str, default="predicts/kalman_gui_windows", help="Output directory for saved windows.")
    parser.add_argument("--array-layout", choices=["time_channel", "channel_time"], default="time_channel", help="Input array layout.")
    parser.add_argument("--fs-hz", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel to load.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of channels to load.")
    parser.add_argument("--time-start", type=int, default=0, help="First time sample to load.")
    parser.add_argument("--time-count", type=int, default=240000, help="Number of time samples to load.")
    parser.add_argument("--window-seconds", type=float, default=240.0, help="Displayed window length in seconds.")
    parser.add_argument("--step-seconds", type=float, default=120.0, help="Step length used by F-key advance and scroll.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    gui = KalmanTrackGUI(
        input_path=args.input,
        out_dir=args.out_dir,
        array_layout=args.array_layout,
        fs_hz=args.fs_hz,
        dx_m=args.dx_m,
        channel_start=args.channel_start,
        channel_count=args.channel_count,
        time_start=args.time_start,
        time_count=args.time_count,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )
    gui.resize(1700, 950)
    gui.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
