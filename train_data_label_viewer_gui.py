"""Visualize simulated DAS training data and `tracks.json` labels.

用途：
    打开由 `simulate_vehicle_sac.py` 生成的训练数据目录，显示 SAC 时空图，
    并叠加 `tracks.json` 中每辆车的真实轨迹点。用于检查训练集里车辆信号
    和标签是否对齐、窗口内有多少辆车、标签点是否覆盖正确通道。

用例：
    uv run python KF/auto_track/train_data_label_viewer_gui.py \
        --data-folder KF/auto_track/datasets/train/sim_0001

输出：
    本脚本启动 PyQt6 图形界面，不写出文件。界面中可切换 Heatmap/Wiggle
    视图、调整窗口长度、拖动窗口起点，并显示当前窗口内的标签轨迹数量。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from matplotlib import cm
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from auto_track_backend import AutoTrackBackend, DEFAULT_DATA_FOLDER


class TrainingLabelViewerGUI(QMainWindow):
    def __init__(self, data_folder: str = DEFAULT_DATA_FOLDER):
        super().__init__()
        self.setWindowTitle("DAS Training Data + Track Label Viewer")

        self.backend = AutoTrackBackend(data_folder=data_folder)
        self.tracks_payload: dict[str, Any] = {}
        self.visible_tracks: list[dict[str, Any]] = []
        self._syncing_slider = False
        self._plot_target_cols = 2600
        self._plot_target_points_per_trace = 4000

        self.fig = Figure(figsize=(11, 6))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        def _tip(widget, text: str) -> None:
            widget.setToolTip(text)
            widget.setStatusTip(text)
            widget.setWhatsThis(text)

        def _label(text: str, tip: str) -> QLabel:
            label = QLabel(text)
            _tip(label, tip)
            return label

        controls = QVBoxLayout()

        path_tip = "训练数据目录，应包含 CH*.sac 和 tracks.json。"
        controls.addWidget(_label("Training Data Folder", path_tip))
        self.path_input = QLineEdit(str(Path(data_folder)))
        self.path_input.setPlaceholderText("Select sim_xxxx folder")
        _tip(self.path_input, path_tip)

        path_buttons = QHBoxLayout()
        self.browse_btn = QPushButton("Browse...")
        self.import_btn = QPushButton("Import Data")
        self.browse_btn.clicked.connect(self.browse_folder)
        self.import_btn.clicked.connect(self.import_data)
        path_buttons.addWidget(self.browse_btn)
        path_buttons.addWidget(self.import_btn)
        controls.addWidget(self.path_input)
        controls.addLayout(path_buttons)

        view_layout = QGridLayout()
        view_layout.setHorizontalSpacing(8)
        view_layout.setVerticalSpacing(6)

        self.view_combo = QComboBox()
        self.view_combo.addItem("Heatmap |data|", "heatmap_abs")
        self.view_combo.addItem("Heatmap raw", "heatmap_raw")
        self.view_combo.addItem("Wiggle traces", "wiggle")
        self.view_combo.currentIndexChanged.connect(lambda _: self.redraw())
        _tip(self.view_combo, "显示方式。Heatmap 更适合直接检查车辆斜线，Wiggle 接近原自动提取界面。")

        self.show_points_check = QCheckBox("Show label points")
        self.show_points_check.setChecked(True)
        self.show_points_check.toggled.connect(lambda _: self.redraw())
        _tip(self.show_points_check, "显示 tracks.json 中每个通道的真实标签点。")

        self.show_ids_check = QCheckBox("Show track IDs")
        self.show_ids_check.setChecked(True)
        self.show_ids_check.toggled.connect(lambda _: self.redraw())
        _tip(self.show_ids_check, "在轨迹中点附近显示 track_id 和速度。")

        self.window_seconds_input = QLineEdit()
        self.window_seconds_input.setPlaceholderText("Window length (s)")
        _tip(self.window_seconds_input, "当前显示窗口长度。")
        self.apply_window_btn = QPushButton("Apply Window Length")
        self.apply_window_btn.clicked.connect(self.apply_window_length)

        view_layout.addWidget(_label("View mode", "显示方式。Heatmap 更适合检查训练数据标签。"), 0, 0)
        view_layout.addWidget(self.view_combo, 0, 1)
        view_layout.addWidget(self.show_points_check, 1, 0, 1, 2)
        view_layout.addWidget(self.show_ids_check, 2, 0, 1, 2)
        view_layout.addWidget(_label("Window seconds", "当前显示窗口长度。"), 3, 0)
        view_layout.addWidget(self.window_seconds_input, 3, 1)
        view_layout.addWidget(self.apply_window_btn, 4, 0, 1, 2)
        controls.addLayout(view_layout)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label)
        controls.addStretch(1)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        controls_widget.setMinimumWidth(330)
        controls_widget.setMaximumWidth(460)

        self.window_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_slider.setMinimum(0)
        self.window_slider.setMaximum(0)
        self.window_slider.setSingleStep(1)
        self.window_slider.setPageStep(1)
        self.window_slider.valueChanged.connect(self.on_window_slider_changed)
        self.window_slider_label = QLabel("Window Start: --")

        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.window_slider_label)
        plot_layout.addWidget(self.window_slider)
        plot_layout.addWidget(self.canvas, stretch=1)
        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(plot_widget, stretch=1)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self._load_labels_from_current_folder(show_errors=False)
        self._sync_window_seconds_input(force=True)
        self._update_window_slider()
        self.redraw()

    def browse_folder(self) -> None:
        start_dir = self.path_input.text().strip() or str(Path.cwd())
        selected = QFileDialog.getExistingDirectory(self, "Select Training Data Folder", start_dir)
        if selected:
            self.path_input.setText(selected)

    def import_data(self) -> None:
        folder = self.path_input.text().strip()
        if not folder:
            QMessageBox.warning(self, "Import Failed", "Please enter a folder path.")
            return
        try:
            self.backend.load_data_folder(folder, reset_results=True)
            self._load_labels_from_current_folder(show_errors=True)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Import Failed", str(exc))
            self.status_label.setText(f"Import failed: {exc}")
            return
        self._sync_window_seconds_input(force=True)
        self._update_window_slider()
        self.redraw()

    def _load_labels_from_current_folder(self, show_errors: bool) -> None:
        label_path = Path(self.backend.files) / "tracks.json"
        self.tracks_payload = {}
        if not label_path.exists():
            msg = f"Imported SAC data, but tracks.json not found: {label_path}"
            self.status_label.setText(msg)
            if show_errors:
                QMessageBox.warning(self, "Labels Missing", msg)
            return
        try:
            self.tracks_payload = json.loads(label_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            msg = f"Cannot read tracks.json: {exc}"
            self.status_label.setText(msg)
            if show_errors:
                QMessageBox.warning(self, "Label Load Failed", msg)
            return
        track_count = len(self.tracks_payload.get("tracks", []))
        self.status_label.setText(f"Imported: {self.backend.files}\nLoaded labels: {track_count} tracks")

    def _sync_window_seconds_input(self, force: bool = False) -> None:
        if self.backend.fs <= 0:
            return
        if (not force) and self.window_seconds_input.hasFocus():
            return
        sec = float(self.backend.window_size) / float(self.backend.fs)
        self.window_seconds_input.setText(f"{sec:.2f}")

    def apply_window_length(self) -> None:
        if self.backend.data_all.size == 0:
            QMessageBox.warning(self, "Notice", "Please import data first.")
            return
        try:
            window_seconds = float(self.window_seconds_input.text().strip())
            if window_seconds <= 0:
                raise ValueError("Window length must be > 0 seconds")
            new_size = int(round(window_seconds * self.backend.fs))
            max_samples = int(self.backend.data_all.shape[1])
            self.backend.window_size = max(1, min(new_size, max_samples))
            self.backend.current_start = min(
                int(self.backend.current_start),
                max(0, max_samples - self.backend.window_size),
            )
            self.backend.update_view_window()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Window Length Setup Failed", str(exc))
            return
        self._sync_window_seconds_input(force=True)
        self._update_window_slider()
        self.redraw()

    def _update_window_slider(self) -> None:
        if self.backend.data_all.size == 0:
            self._syncing_slider = True
            self.window_slider.setRange(0, 0)
            self.window_slider.setValue(0)
            self._syncing_slider = False
            self.window_slider.setEnabled(False)
            self.window_slider_label.setText("Window Start: --")
            return

        max_start = max(0, int(self.backend.data_all.shape[1] - self.backend.window_size))
        cur = int(max(0, min(self.backend.current_start, max_start)))
        step = int(max(1, min(self.backend.scroll_step_samples, max(1, self.backend.window_size // 5))))
        page = int(max(1, self.backend.window_size // 2))

        self._syncing_slider = True
        self.window_slider.setRange(0, max_start)
        self.window_slider.setSingleStep(step)
        self.window_slider.setPageStep(page)
        self.window_slider.setValue(cur)
        self._syncing_slider = False
        self.window_slider.setEnabled(max_start > 0)

        t0 = cur / self.backend.fs
        t1 = (cur + self.backend.window_size) / self.backend.fs
        self.window_slider_label.setText(f"Window Start: {t0:.1f} s   End: {t1:.1f} s")

    def on_window_slider_changed(self, value: int) -> None:
        if self._syncing_slider or self.backend.data_all.size == 0:
            return
        max_start = max(0, int(self.backend.data_all.shape[1] - self.backend.window_size))
        new_start = int(max(0, min(int(value), max_start)))
        if new_start == int(self.backend.current_start):
            return
        self.backend.current_start = new_start
        self.backend.update_view_window()
        self.redraw()

    def _plot_heatmap(self, mode: str) -> None:
        data = np.asarray(self.backend.data_view, dtype=np.float32)
        if data.size == 0:
            return
        decim = int(max(1, np.ceil(data.shape[1] / float(self._plot_target_cols))))
        plot = data[:, ::decim]
        if mode == "heatmap_abs":
            shown = np.abs(plot)
            clip = float(max(np.percentile(shown[np.isfinite(shown)], 99.5), 1e-6))
            cmap_name = "magma"
            vmin, vmax = 0.0, clip
            shown = np.clip(shown, vmin, vmax)
        else:
            finite = plot[np.isfinite(plot)]
            clip = float(max(np.percentile(np.abs(finite), 99.5), 1e-6)) if finite.size else 1.0
            cmap_name = "seismic"
            vmin, vmax = -clip, clip
            shown = np.clip(plot, vmin, vmax)

        x0 = float(self.backend.x_axis_m[0]) * 1e-3
        x1 = float(self.backend.x_axis_m[-1]) * 1e-3
        t0 = 0.0
        t1 = float(data.shape[1] - 1) / float(self.backend.fs)
        self.ax.imshow(
            shown.T,
            aspect="auto",
            origin="lower",
            cmap=cmap_name,
            extent=[x0, x1, t0, t1],
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

    def _plot_wiggle(self) -> None:
        data = self.backend.data_view
        if data.size == 0:
            return
        n_ch, n_samples = data.shape
        decim = int(max(1, np.ceil(n_samples / float(self._plot_target_points_per_trace))))
        t = self.backend.t_axis_view[::decim]
        data_plot = np.asarray(data[:, ::decim], dtype=np.float64)
        offsets_km = np.asarray(self.backend.x_axis_m, dtype=np.float64) * 1e-3
        spacing = float(np.median(np.diff(offsets_km))) if offsets_km.size >= 2 else 0.1
        if not np.isfinite(spacing) or spacing <= 0:
            spacing = 0.1
        wiggle_amp = 0.27 * spacing
        abs_vals = np.abs(data_plot[np.isfinite(data_plot)])
        global_ref = max(float(np.quantile(abs_vals, 0.995)), 1e-12) if abs_vals.size else 1.0
        for i in range(n_ch):
            ratio = np.clip(data_plot[i] / global_ref, -1.35, 1.35)
            x = offsets_km[i] + ratio * wiggle_amp
            self.ax.plot(x, t, color="0.45", linewidth=0.8, alpha=0.85)

    def _visible_label_tracks(self) -> list[dict[str, Any]]:
        start = int(self.backend.current_start)
        end = int(self.backend.current_start + self.backend.window_size)
        visible = []
        for track in self.tracks_payload.get("tracks", []):
            pts = [
                p
                for p in track.get("points", [])
                if start <= int(p.get("t_idx", -1)) < end
            ]
            if not pts:
                continue
            visible.append({"track": track, "points": pts})
        return visible

    def _plot_labels(self) -> None:
        visible = self._visible_label_tracks()
        self.visible_tracks = visible
        if not visible:
            return
        cmap = cm.get_cmap("tab20", max(1, len(visible)))
        t0 = int(self.backend.current_start)
        fs = float(self.backend.fs)
        x_axis = np.asarray(self.backend.x_axis_m, dtype=np.float64)
        for idx, item in enumerate(visible):
            track = item["track"]
            points = sorted(item["points"], key=lambda p: int(p.get("ch_idx", 0)))
            xs = []
            ts = []
            for p in points:
                ch = int(p.get("ch_idx", 0))
                if 0 <= ch < x_axis.size:
                    x_m = float(p.get("offset_m", x_axis[ch]))
                else:
                    x_m = float(p.get("offset_m", ch * self.backend.dx_m))
                xs.append(x_m * 1e-3)
                ts.append((int(p.get("t_idx", 0)) - t0) / fs)
            if len(xs) == 0:
                continue
            color = cmap(idx % max(1, cmap.N))
            self.ax.plot(xs, ts, color=color, linewidth=1.6, alpha=0.95)
            if self.show_points_check.isChecked():
                self.ax.scatter(xs, ts, color=[color], s=12, marker="o", alpha=0.95)
            if self.show_ids_check.isChecked():
                mid = len(xs) // 2
                label = f"#{track.get('track_id', idx)}"
                speed = track.get("speed_kmh")
                try:
                    speed_float = float(speed)
                    if np.isfinite(speed_float):
                        label += f" {speed_float:.0f}km/h"
                except Exception:  # noqa: BLE001
                    pass
                self.ax.text(
                    xs[mid],
                    ts[mid],
                    label,
                    color=color,
                    fontsize=8,
                    ha="left",
                    va="center",
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.8},
                )

    def redraw(self) -> None:
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        if self.backend.data_view.size == 0:
            self.ax.set_axis_off()
            self.ax.text(
                0.5,
                0.5,
                "No data loaded\nPlease import a training data folder first",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            self._update_window_slider()
            self._sync_window_seconds_input()
            self.canvas.draw_idle()
            return

        mode = str(self.view_combo.currentData())
        if mode == "wiggle":
            self._plot_wiggle()
        else:
            self._plot_heatmap(mode)
        self._plot_labels()

        y0 = 0.0
        y1 = float(self.backend.data_view.shape[1] - 1) / float(self.backend.fs)
        self.ax.set_ylim(y0, y1)
        self.ax.invert_yaxis()

        x0 = float(self.backend.x_axis_m[0]) * 1e-3
        x1 = float(self.backend.x_axis_m[-1]) * 1e-3
        if self.backend.x_axis_m.size >= 2:
            dx = float(np.median(np.diff(self.backend.x_axis_m))) * 1e-3
        else:
            dx = 0.1
        pad = 0.4 * dx if np.isfinite(dx) and dx > 0 else 0.04
        self.ax.set_xlim(x0 - pad, x1 + pad)
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y + self.backend.t_offset:.1f}"))
        self.ax.set_xlabel("Offset [km]")
        self.ax.set_ylabel("Time [s]")
        self.ax.set_title(
            f"Visible label tracks={len(self.visible_tracks)}, "
            f"Window=[{self.backend.t_offset:.1f}, {self.backend.t_offset + y1:.1f}] s"
        )
        self.status_label.setText(
            f"Imported: {self.backend.files}\n"
            f"Total labels: {len(self.tracks_payload.get('tracks', []))}, "
            f"visible in window: {len(self.visible_tracks)}"
        )
        self._sync_window_seconds_input()
        self._update_window_slider()
        self.canvas.draw_idle()

    def on_scroll(self, event) -> None:
        if event is None or self.backend.data_all.size == 0:
            return
        if getattr(self.toolbar, "mode", ""):
            return
        if hasattr(event, "inaxes") and event.inaxes is not self.ax:
            return
        step = getattr(event, "step", 0)
        if step == 0:
            step = 1 if getattr(event, "button", None) == "up" else -1
        step = 1 if step > 0 else -1
        changed = self.backend.handle_scroll(step, zoom=False, cursor_x=None)
        if changed:
            self.redraw()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize DAS training SAC data with tracks.json labels.")
    parser.add_argument(
        "--data-folder",
        type=str,
        default=DEFAULT_DATA_FOLDER,
        help="Training data folder containing CH*.sac and tracks.json.",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    gui = TrainingLabelViewerGUI(data_folder=args.data_folder)
    gui.resize(1700, 900)
    gui.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
