"""Labeling GUI built on top of the original auto-track GUI.

用途：
    直接复用原来的 `AutoTrackGUI` 界面、参数和图搜索提取流程，在其基础上
    增加真实数据打标签和手动校准能力。这样自动提取部分和原 GUI 保持同一套
    行为，只额外增加标签工程、点级编辑和标签导出。

用例：
    uv run python -m autotrack.gui.real_data_label_gui \
        --data-folder /Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy

标签工作流：
    1. 用原 GUI 的导入和参数面板配置图搜索参数；
    2. 点击 `Graph Auto Label Current Window`；
    3. 在右侧选择轨迹，切换编辑模式做人工修订；
    4. 保存 `manual_labels.json`，或导出 `manual_labels.csv`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.backend_bases import MouseButton
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QWidget,
)

from autotrack.core.auto_track_backend import DEFAULT_DATA_FOLDER
from autotrack.core.track_extractor_graph import ExtractorConfig, _build_nodes
from autotrack.gui.auto_track_gui import AutoTrackGUI
from autotrack.kalman import KFTracking, build_default_args
from autotrack.labeling import TrackLabelProject


matplotlib.rcParams["font.family"] = "Times New Roman"


class RealDataLabelGUI(AutoTrackGUI):
    def __init__(self, data_folder: str = DEFAULT_DATA_FOLDER):
        self.project: Optional[TrackLabelProject] = None
        self.project_path: Optional[str] = None
        self.selected_track_id: Optional[int] = None
        self._adopt_extract_on_finish = False
        super().__init__(data_folder=data_folder)
        self.setWindowTitle("Vehicle Trajectory Labeling GUI (Original Graph GUI + Label Tools)")

        self.project = TrackLabelProject(
            source_path=self.backend.files,
            source_kind=str(self.backend.last_import_info.get("input_kind", "unknown")),
            fs_hz=float(self.backend.fs),
            dx_m=float(self.backend.dx_m),
        )

        self.canvas.mpl_connect("button_press_event", self.on_plot_click)
        self._augment_layout_with_label_panel()
        self._refresh_track_list()
        self.redraw()

    @staticmethod
    def _track_is_saved(track) -> bool:
        return str(getattr(track, "note", "") or "").strip().startswith("[saved]")

    def _augment_layout_with_label_panel(self) -> None:
        root = self.centralWidget()
        if root is None or root.layout() is None:
            return
        main_layout = root.layout()

        self.label_panel = QWidget()
        self.label_panel.setMinimumWidth(300)
        self.label_panel.setMaximumWidth(420)
        panel_layout = QVBoxLayout()

        title = QLabel("Label Tools")
        panel_layout.addWidget(title)

        button_grid = QGridLayout()
        self.graph_auto_label_btn = QPushButton("Graph Auto Label Current Window")
        self.graph_auto_label_btn.clicked.connect(self.graph_auto_label_current_window)
        self.adopt_current_extract_btn = QPushButton("Adopt Current Extract Result")
        self.adopt_current_extract_btn.clicked.connect(self.adopt_current_extract_result)
        self.clear_window_labels_btn = QPushButton("Clear Window Labels")
        self.clear_window_labels_btn.clicked.connect(self.clear_current_window_labels)
        self.clear_all_labels_btn = QPushButton("Clear All Labels")
        self.clear_all_labels_btn.clicked.connect(self.clear_all_labels)
        button_grid.addWidget(self.graph_auto_label_btn, 0, 0, 1, 2)
        button_grid.addWidget(self.adopt_current_extract_btn, 1, 0, 1, 2)
        button_grid.addWidget(self.clear_window_labels_btn, 2, 0)
        button_grid.addWidget(self.clear_all_labels_btn, 2, 1)
        panel_layout.addLayout(button_grid)

        edit_grid = QGridLayout()
        self.click_backend_combo = QComboBox()
        self.click_backend_combo.addItem("Graph", "graph")
        self.click_backend_combo.addItem("Kalman", "kalman")
        self.edit_mode_combo = QComboBox()
        self.edit_mode_combo.addItem("Select track", "select")
        self.edit_mode_combo.addItem("Add / move point", "add_move")
        self.edit_mode_combo.addItem("Delete point", "delete_point")
        self.edit_mode_combo.addItem("Extend graph search", "extend_graph")
        self.new_track_btn = QPushButton("New Track")
        self.new_track_btn.clicked.connect(self.create_track)
        self.delete_track_btn = QPushButton("Delete Selected Track")
        self.delete_track_btn.clicked.connect(self.delete_selected_track)
        edit_grid.addWidget(QLabel("Click backend"), 0, 0)
        edit_grid.addWidget(self.click_backend_combo, 0, 1)
        edit_grid.addWidget(QLabel("Edit Mode"), 1, 0)
        edit_grid.addWidget(self.edit_mode_combo, 1, 1)
        edit_grid.addWidget(self.new_track_btn, 2, 0)
        edit_grid.addWidget(self.delete_track_btn, 2, 1)
        panel_layout.addLayout(edit_grid)

        panel_layout.addWidget(QLabel("Visible Label Tracks"))
        self.track_list = QListWidget()
        self.track_list.currentItemChanged.connect(self.on_track_list_changed)
        panel_layout.addWidget(self.track_list, stretch=1)

        io_grid = QGridLayout()
        self.load_labels_btn = QPushButton("Load Labels JSON")
        self.load_labels_btn.clicked.connect(self.load_labels_json)
        self.save_labels_btn = QPushButton("Save Labels JSON")
        self.save_labels_btn.clicked.connect(self.save_labels_json)
        self.export_labels_btn = QPushButton("Export Label CSV")
        self.export_labels_btn.clicked.connect(self.export_label_csv)
        self.save_window_bundle_btn = QPushButton("Save Current Window + Labels")
        self.save_window_bundle_btn.clicked.connect(self.save_current_window_bundle)
        io_grid.addWidget(self.load_labels_btn, 0, 0)
        io_grid.addWidget(self.save_labels_btn, 0, 1)
        io_grid.addWidget(self.export_labels_btn, 1, 0, 1, 2)
        io_grid.addWidget(self.save_window_bundle_btn, 2, 0, 1, 2)
        panel_layout.addLayout(io_grid)

        guide = QLabel(
            "Click behavior:\n"
            "- Select track: click a label point\n"
            "- Add / move point: click plot to write one point on nearest channel\n"
            "- Delete point: click near a point on selected track\n"
            "- Kalman backend only extends the selected graph track outside its existing points\n"
            "- Press S: mark selected track as saved and deselect"
        )
        guide.setWordWrap(True)
        panel_layout.addWidget(guide)

        self.label_panel.setLayout(panel_layout)
        main_layout.addWidget(self.label_panel)

    def _current_window_bounds(self) -> tuple[int, int]:
        start = int(self.backend.current_start)
        end = int(start + self.backend.window_size)
        return start, end

    def _refresh_project_source(self) -> None:
        if self.project is None:
            return
        self.project.load_source_defaults(
            source_path=self.backend.files,
            source_kind=str(self.backend.last_import_info.get("input_kind", "unknown")),
            fs_hz=float(self.backend.fs),
            dx_m=float(self.backend.dx_m),
        )

    def _refresh_track_list(self) -> None:
        if self.project is None or (not hasattr(self, "track_list")):
            return
        start, end = self._current_window_bounds()
        visible = self.project.visible_tracks(start, end)
        self.track_list.blockSignals(True)
        self.track_list.clear()
        current_row = -1
        for idx, track in enumerate(visible):
            saved_prefix = "[S] " if self._track_is_saved(track) else ""
            speed_text = (
                f"{track.mean_speed_kmh:.1f} km/h"
                if np.isfinite(track.mean_speed_kmh)
                else "nan"
            )
            item = QListWidgetItem(
                f"{saved_prefix}#{track.track_id}  {track.direction}  pts={len(track.points)}  speed={speed_text}"
            )
            item.setData(Qt.ItemDataRole.UserRole, int(track.track_id))
            self.track_list.addItem(item)
            if self.selected_track_id is not None and int(track.track_id) == int(self.selected_track_id):
                current_row = idx
        if current_row >= 0:
            self.track_list.setCurrentRow(current_row)
        self.track_list.blockSignals(False)

    def graph_auto_label_current_window(self) -> None:
        self._adopt_extract_on_finish = True
        self.start_extract(current_window_only=True)

    def adopt_current_extract_result(self) -> None:
        if self.project is None:
            return
        if not self.backend.tracks:
            QMessageBox.information(self, "No Extract Result", "Please run extraction first.")
            return
        start, end = self._current_window_bounds()
        tracks = TrackLabelProject.from_backend_tracks(self.backend.tracks, source="auto")
        self.project.replace_tracks_in_window(start_sample=start, end_sample=end, tracks=tracks)
        self._refresh_track_list()
        self.redraw()
        self.status_label.setText(f"Adopted current extraction result into labels: {len(tracks)} tracks")

    def clear_current_window_labels(self) -> None:
        if self.project is None:
            return
        start, end = self._current_window_bounds()
        removed = self.project.remove_tracks_in_window(start, end)
        if self.selected_track_id is not None and self.project.get_track(self.selected_track_id) is None:
            self.selected_track_id = None
        self._refresh_track_list()
        self.redraw()
        self.status_label.setText(f"Cleared current-window labels, affected tracks: {removed}")

    def clear_all_labels(self) -> None:
        if self.project is None:
            return
        self.project.tracks = []
        self.project.touch()
        self.selected_track_id = None
        self._refresh_track_list()
        self.redraw()
        self.status_label.setText("All labels cleared")

    def create_track(self) -> None:
        if self.project is None:
            return
        track = self.project.add_track(direction=str(self.direction_combo.currentData()), source="manual")
        self.project.renumber_tracks()
        self.selected_track_id = int(track.track_id)
        self._refresh_track_list()
        self.redraw()
        self.status_label.setText(f"Created track #{self.selected_track_id}")

    def delete_selected_track(self) -> None:
        if self.project is None:
            return
        if self.selected_track_id is None:
            QMessageBox.information(self, "Delete Track", "Please select a label track first.")
            return
        if self.project.delete_track(self.selected_track_id):
            self.selected_track_id = None
            self._refresh_track_list()
            self.redraw()
            self.status_label.setText("Deleted selected label track")

    def mark_selected_track_saved(self) -> None:
        if self.project is None or self.selected_track_id is None:
            return
        track = self.project.get_track(self.selected_track_id)
        if track is None:
            return
        note = str(track.note or "").strip()
        if not note.startswith("[saved]"):
            track.note = "[saved]" if not note else f"[saved] {note}"
        self.project.touch()
        saved_track_id = int(track.track_id)
        self.selected_track_id = None
        self._refresh_track_list()
        self.redraw()
        self.status_label.setText(
            f"Marked track #{saved_track_id} as saved. Right-click can start a new trajectory."
        )

    def on_track_list_changed(self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]) -> None:
        if current is None:
            return
        track_id = current.data(Qt.ItemDataRole.UserRole)
        if track_id is not None:
            self.selected_track_id = int(track_id)
            self.redraw()

    def load_labels_json(self) -> None:
        start_dir = self.project_path or self.path_input.text().strip() or str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Load Label Project JSON",
            start_dir,
            "Label Project (*.json);;All Files (*)",
        )
        if not selected:
            return
        try:
            self.project = TrackLabelProject.from_json(selected)
            self.project_path = str(Path(selected).expanduser())
            self.selected_track_id = None
            self._refresh_track_list()
            self.redraw()
            self.status_label.setText(f"Loaded labels: {selected}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Failed", str(exc))

    def save_labels_json(self) -> None:
        if self.project is None:
            return
        self._refresh_project_source()
        default_path = self.project_path or str(Path(self.backend.files).with_name("manual_labels.json"))
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Label Project JSON",
            default_path,
            "Label Project (*.json);;All Files (*)",
        )
        if not out_path:
            return
        try:
            self.project_path = self.project.save_json(out_path)
            self.status_label.setText(f"Saved labels: {self.project_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Failed", str(exc))

    def export_label_csv(self) -> None:
        if self.project is None:
            return
        default_path = str(Path(self.backend.files).with_name("manual_labels.csv"))
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Label CSV",
            default_path,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not out_path:
            return
        try:
            csv_path = self.project.export_csv(out_path)
            self.status_label.setText(f"Exported label CSV: {csv_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export Failed", str(exc))

    def _window_export_dir(self) -> str:
        source_path = Path(self.backend.files).expanduser()
        base_dir = source_path.parent if source_path.suffix else source_path
        return str(base_dir / "labeled_windows")

    def save_current_window_bundle(self) -> None:
        if self.project is None:
            return
        start, end = self._current_window_bounds()
        visible_tracks = self.project.visible_tracks(start, end)
        if not visible_tracks:
            QMessageBox.information(self, "Save Current Window", "Current window has no visible labels.")
            return

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder For Current Window",
            self._window_export_dir(),
        )
        if not out_dir:
            return

        crop = np.asarray(self.backend.data_view, dtype=np.float32)
        ch0 = 0
        ch1 = int(self.backend.data_all.shape[0] - 1)
        t0 = int(start)
        t1 = int(end - 1)

        bundle_name = (
            f"window_t{t0:09d}_t{t1:09d}_"
            f"ch{ch0:04d}_{ch1:04d}_tracks{len(visible_tracks):03d}"
        )
        bundle_dir = Path(out_dir).expanduser() / bundle_name
        bundle_dir.mkdir(parents=True, exist_ok=True)

        npy_path = bundle_dir / "data.npy"
        meta_path = bundle_dir / "meta.json"
        labels_json_path = bundle_dir / "labels.json"
        labels_csv_path = bundle_dir / "labels.csv"

        np.save(str(npy_path), crop)

        local_tracks_payload = []
        for track in visible_tracks:
            payload = dict(self.project.track_payload(track))
            payload["points"] = [
                {
                    **point,
                    "local_ch_idx": int(point["ch_idx"]) - int(ch0),
                    "local_t_idx": int(point["t_idx"]) - int(t0),
                    "local_time_s": (int(point["t_idx"]) - int(t0)) / float(self.backend.fs),
                }
                for point in payload["points"]
                if start <= int(point["t_idx"]) < end
            ]
            if payload["points"]:
                local_tracks_payload.append(payload)

        labels_json_path.write_text(
            json.dumps(
                {
                    "source_path": self.backend.files,
                    "source_kind": str(self.backend.last_import_info.get("input_kind", "unknown")),
                    "fs_hz": float(self.backend.fs),
                    "dx_m": float(self.backend.dx_m),
                    "crop_ch_start": int(ch0),
                    "crop_ch_end": int(ch1),
                    "crop_t_start": int(t0),
                    "crop_t_end": int(t1),
                    "track_count": int(len(local_tracks_payload)),
                    "tracks": local_tracks_payload,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        meta_path.write_text(
            json.dumps(
                {
                    "source_path": self.backend.files,
                    "source_kind": str(self.backend.last_import_info.get("input_kind", "unknown")),
                    "fs_hz": float(self.backend.fs),
                    "dx_m": float(self.backend.dx_m),
                    "array_shape": [int(crop.shape[0]), int(crop.shape[1])],
                    "crop_ch_start": int(ch0),
                    "crop_ch_end": int(ch1),
                    "crop_t_start": int(t0),
                    "crop_t_end": int(t1),
                    "window_seconds": float(self.backend.window_size) / float(self.backend.fs),
                    "track_count": int(len(local_tracks_payload)),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.project.export_csv(labels_csv_path)
        rows = labels_csv_path.read_text(encoding="utf-8").splitlines()
        if rows:
            header = rows[0]
            visible_ids = {int(track.track_id) for track in visible_tracks}
            filtered = [header]
            for row in rows[1:]:
                if not row:
                    continue
                try:
                    track_id = int(row.split(",", 1)[0])
                except Exception:  # noqa: BLE001
                    continue
                if track_id in visible_ids:
                    filtered.append(row)
            labels_csv_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")

        self.status_label.setText(
            f"Saved current-window bundle: {bundle_dir}\n"
            f"data.npy shape={tuple(crop.shape)}, tracks={len(local_tracks_payload)}"
        )

    def _nearest_channel_index(self, x_km: float) -> int:
        x_m = float(x_km) * 1000.0
        idx = int(np.argmin(np.abs(np.asarray(self.backend.x_axis_m, dtype=np.float64) - x_m)))
        return max(0, min(idx, int(self.backend.x_axis_m.size - 1)))

    @staticmethod
    def _is_right_click(event) -> bool:
        button = getattr(event, "button", None)
        if button == MouseButton.RIGHT:
            return True
        return str(button).lower() in {"mousebutton.right", "3", "right"}

    def _click_backend(self) -> str:
        return str(self.click_backend_combo.currentData()) if hasattr(self, "click_backend_combo") else "graph"

    def _graph_config_from_gui(self) -> ExtractorConfig:
        return ExtractorConfig(
            use_template_enhancement=bool(self.template_enhance_check.isChecked()),
            prominence=float(self.prominence_input.text().strip()),
            min_peak_distance=int(float(self.min_peak_distance_input.text().strip())),
            min_track_channels=int(float(self.min_track_channels_input.text().strip())),
            edge_min_track_channels=int(float(self.edge_min_track_channels_input.text().strip())),
            edge_time_margin_seconds=float(self.edge_time_margin_seconds_input.text().strip()),
            edge_min_score_scale=float(self.edge_min_score_scale_input.text().strip()),
        )

    @staticmethod
    def _track_point_distance_cost(point, click_ch_idx: int, click_t_idx: int) -> float:
        dch = abs(int(point.ch_idx) - int(click_ch_idx))
        dt = abs(int(point.t_idx) - int(click_t_idx))
        return float(dch) + 0.0025 * float(dt)

    def _snap_channel_time_to_peak(
        self,
        nodes: list[dict[str, np.ndarray]],
        ch_idx: int,
        guide_t_idx: int,
        search_radius: int,
        continuity_t_idx: Optional[int] = None,
    ) -> Optional[tuple[int, float, float]]:
        t_arr = np.asarray(nodes[ch_idx]["t"], dtype=np.int32)
        amp_arr = np.asarray(nodes[ch_idx]["amp"], dtype=np.float32)
        score_arr = np.asarray(nodes[ch_idx]["score"], dtype=np.float32)

        lo = max(0, int(guide_t_idx) - int(search_radius))
        hi = min(int(self.backend.data_all.shape[1] - 1), int(guide_t_idx) + int(search_radius))
        valid = (t_arr >= lo) & (t_arr <= hi)
        if np.any(valid):
            idxs = np.where(valid)[0]
            best_idx = -1
            best_cost = float("inf")
            cont_ref = int(continuity_t_idx) if continuity_t_idx is not None else int(guide_t_idx)
            cont_scale = max(60.0, 0.5 * float(search_radius))
            guide_scale = max(60.0, 0.5 * float(search_radius))
            for idx in idxs:
                t_idx = int(t_arr[idx])
                guide_cost = abs(float(t_idx - guide_t_idx)) / guide_scale
                cont_cost = abs(float(t_idx - cont_ref)) / cont_scale
                score_bonus = 0.18 * float(score_arr[idx])
                cost = guide_cost + 0.55 * cont_cost - score_bonus
                if cost < best_cost:
                    best_cost = cost
                    best_idx = int(idx)
            if best_idx >= 0:
                return int(t_arr[best_idx]), float(amp_arr[best_idx]), float(score_arr[best_idx])

        return None

    def _trace_from_seed_one_side(
        self,
        *,
        nodes: list[dict[str, np.ndarray]],
        seed_ch_idx: int,
        seed_t_idx: int,
        step_sign: int,
        direction: str,
        vmin_kmh: float,
        vmax_kmh: float,
        search_radius_base: int,
    ) -> list[tuple[int, int, float, float]]:
        fs = float(self.backend.fs)
        dx_m = float(self.backend.dx_m)
        vmin_mps = float(vmin_kmh) / 3.6
        vmax_mps = float(vmax_kmh) / 3.6
        speed_ref = 0.5 * (vmin_mps + vmax_mps)
        prev_ch = int(seed_ch_idx)
        prev_t = int(seed_t_idx)
        traced: list[tuple[int, int, float, float]] = []
        max_skip = max(6, int(float(self.min_track_channels_input.text().strip()) // 2))

        while True:
            best: Optional[tuple[int, int, float, float, float]] = None
            best_cost = float("inf")
            found_any_channel = False

            for dch in range(1, max_skip + 1):
                ch_idx = int(prev_ch) + int(step_sign) * int(dch)
                if ch_idx < 0 or ch_idx >= int(self.backend.data_all.shape[0]):
                    continue
                found_any_channel = True
                delta_x = float(abs(ch_idx - prev_ch) * dx_m)
                dt_low, dt_high = self._dt_bounds_for_trace(
                    direction=direction,
                    step_sign=int(step_sign),
                    delta_x_m=delta_x,
                    vmin_mps=vmin_mps,
                    vmax_mps=vmax_mps,
                )
                pred_dt = self._pred_dt_for_trace(
                    direction=direction,
                    step_sign=int(step_sign),
                    delta_x_m=delta_x,
                    speed_mps=float(speed_ref),
                )
                pred_t_idx = int(round(float(prev_t) + pred_dt * fs))
                snapped = self._snap_channel_time_to_peak(
                    nodes=nodes,
                    ch_idx=int(ch_idx),
                    guide_t_idx=int(pred_t_idx),
                    search_radius=int(max(search_radius_base, abs(pred_dt) * fs * 0.6 + 40.0)),
                    continuity_t_idx=int(pred_t_idx),
                )
                if snapped is None:
                    continue
                cand_t_idx, cand_amp, cand_score = snapped
                dt = (float(cand_t_idx) - float(prev_t)) / fs
                dt_lo = min(float(dt_low), float(dt_high))
                dt_hi = max(float(dt_low), float(dt_high))
                if dt < dt_lo or dt > dt_hi:
                    continue
                speed_curr = float(delta_x / max(abs(dt), 1e-9))
                cost = (
                    abs(float(cand_t_idx) - float(pred_t_idx)) / max(80.0, float(search_radius_base))
                    + 0.18 * max(0, dch - 1)
                    - 0.08 * float(cand_score)
                )
                if cost < best_cost:
                    best_cost = cost
                    best = (int(ch_idx), int(cand_t_idx), float(cand_amp), float(cand_score), float(speed_curr))

            if best is None:
                if found_any_channel:
                    break
                return traced

            ch_idx, cand_t_idx, cand_amp, cand_score, speed_curr = best
            traced.append((ch_idx, cand_t_idx, cand_amp, cand_score))
            prev_ch = int(ch_idx)
            prev_t = int(cand_t_idx)
            speed_ref = 0.65 * float(speed_ref) + 0.35 * float(speed_curr)

        return traced

    @staticmethod
    def _dt_bounds_for_trace(
        direction: str,
        step_sign: int,
        delta_x_m: float,
        vmin_mps: float,
        vmax_mps: float,
    ) -> tuple[float, float]:
        if direction == "forward":
            sign = 1.0 if int(step_sign) > 0 else -1.0
        elif direction == "reverse":
            sign = -1.0 if int(step_sign) > 0 else 1.0
        else:
            raise ValueError("direction must be forward or reverse")

        lo = sign * (delta_x_m / vmax_mps)
        hi = sign * (delta_x_m / vmin_mps)
        return (min(lo, hi), max(lo, hi))

    @staticmethod
    def _pred_dt_for_trace(
        direction: str,
        step_sign: int,
        delta_x_m: float,
        speed_mps: float,
    ) -> float:
        base = float(delta_x_m) / max(float(speed_mps), 1e-9)
        if direction == "forward":
            return base if int(step_sign) > 0 else -base
        if direction == "reverse":
            return -base if int(step_sign) > 0 else base
        raise ValueError("direction must be forward or reverse")

    def _create_track_from_right_click_peak(self, click_ch_idx: int, click_t_idx: int) -> int:
        cfg = self._graph_config_from_gui()
        nodes = _build_nodes(np.asarray(self.backend.data_all, dtype=np.float32), float(self.backend.fs), cfg)
        search_radius = int(max(120, float(self.min_peak_distance_input.text().strip()) * 0.5))
        seed = self._snap_channel_time_to_peak(
            nodes=nodes,
            ch_idx=int(click_ch_idx),
            guide_t_idx=int(click_t_idx),
            search_radius=int(search_radius),
            continuity_t_idx=int(click_t_idx),
        )
        if seed is None:
            return 0

        seed_t_idx, seed_amp, seed_score = seed
        direction = str(self.direction_combo.currentData())
        vmin_kmh = float(self.speed_min_input.text().strip())
        vmax_kmh = float(self.speed_max_input.text().strip())

        left = self._trace_from_seed_one_side(
            nodes=nodes,
            seed_ch_idx=int(click_ch_idx),
            seed_t_idx=int(seed_t_idx),
            step_sign=-1,
            direction=direction,
            vmin_kmh=vmin_kmh,
            vmax_kmh=vmax_kmh,
            search_radius_base=int(search_radius),
        )
        right = self._trace_from_seed_one_side(
            nodes=nodes,
            seed_ch_idx=int(click_ch_idx),
            seed_t_idx=int(seed_t_idx),
            step_sign=1,
            direction=direction,
            vmin_kmh=vmin_kmh,
            vmax_kmh=vmax_kmh,
            search_radius_base=int(search_radius),
        )

        if self.project is None:
            return 0
        track = self.project.add_track(direction=direction, source="right_click_seed")
        self.project.upsert_point(
            track_id=int(track.track_id),
            ch_idx=int(click_ch_idx),
            t_idx=int(seed_t_idx),
            offset_m=float(self.backend.x_axis_m[int(click_ch_idx)]),
            amp=float(seed_amp),
            score=float(seed_score),
            source="right_click_seed",
        )
        for ch_idx, t_idx, amp, score in left + right:
            self.project.upsert_point(
                track_id=int(track.track_id),
                ch_idx=int(ch_idx),
                t_idx=int(t_idx),
                offset_m=float(self.backend.x_axis_m[int(ch_idx)]),
                amp=float(amp),
                score=float(score),
                source="right_click_seed",
            )
        self.project.renumber_tracks()
        self.selected_track_id = int(track.track_id)
        return 1 + len(left) + len(right)

    def _kalman_trace_one_side(
        self,
        *,
        seed_ch_idx: int,
        seed_t_idx: int,
        step_sign: int,
        direction: str,
        stop_ch_idx: Optional[int] = None,
        sigma_a: float = 0.0001,
    ) -> list[tuple[int, int, float, float]]:
        data_all = np.asarray(self.backend.data_all, dtype=np.float32)
        n_ch = int(data_all.shape[0])
        if int(step_sign) > 0:
            ch_slice = data_all[int(seed_ch_idx) :, :]
            physical_channels = list(range(int(seed_ch_idx), n_ch))
            slope_sign = 1.0 if direction == "forward" else -1.0
        else:
            ch_slice = data_all[: int(seed_ch_idx) + 1, :][::-1, :]
            physical_channels = list(range(int(seed_ch_idx), -1, -1))
            slope_sign = -1.0 if direction == "forward" else 1.0

        if ch_slice.shape[0] <= 1:
            return []

        local_x = np.arange(ch_slice.shape[0], dtype=np.float64) * float(self.backend.dx_m)
        local_t = np.arange(ch_slice.shape[1], dtype=np.float64) / float(self.backend.fs)
        kf_args = build_default_args()
        veh_args = dict(kf_args["veh"])
        veh_args["vel_init"] = 25.0 * slope_sign
        if slope_sign > 0:
            veh_args["vmin"] = 20.0
            veh_args["vmax"] = 30.0
        else:
            veh_args["vmin"] = -30.0
            veh_args["vmax"] = -20.0

        tracker = KFTracking(ch_slice, local_t, local_x, kf_args)
        tracked_states, _weak_states = tracker.tracking_with_veh_base(
            start_x=float(local_x[0]),
            end_x=float(local_x[-1]),
            veh_base=np.array([int(seed_t_idx)], dtype=np.float64),
            sigma_a=float(sigma_a),
            veh_args=veh_args,
        )
        states = np.asarray(tracked_states[0], dtype=np.float64)
        traced: list[tuple[int, int, float, float]] = []
        for local_idx, value in enumerate(states):
            if np.isnan(value):
                continue
            physical_ch = int(physical_channels[int(local_idx)])
            if stop_ch_idx is not None:
                if int(step_sign) > 0 and physical_ch > int(stop_ch_idx):
                    break
                if int(step_sign) < 0 and physical_ch < int(stop_ch_idx):
                    break
            t_idx = int(np.clip(int(round(float(value))), 0, data_all.shape[1] - 1))
            amp = float(data_all[physical_ch, t_idx])
            traced.append((physical_ch, t_idx, amp, 1.0))
        if traced and traced[0][0] == int(seed_ch_idx):
            traced = traced[1:]
        return traced

    def _create_track_from_right_click_peak_kalman(self, click_ch_idx: int, click_t_idx: int) -> int:
        cfg = self._graph_config_from_gui()
        nodes = _build_nodes(np.asarray(self.backend.data_all, dtype=np.float32), float(self.backend.fs), cfg)
        search_radius = int(max(120, float(self.min_peak_distance_input.text().strip()) * 0.5))
        seed = self._snap_channel_time_to_peak(
            nodes=nodes,
            ch_idx=int(click_ch_idx),
            guide_t_idx=int(click_t_idx),
            search_radius=int(search_radius),
            continuity_t_idx=int(click_t_idx),
        )
        if seed is None or self.project is None:
            return 0

        seed_t_idx, seed_amp, seed_score = seed
        direction = str(self.direction_combo.currentData())
        left = self._kalman_trace_one_side(
            seed_ch_idx=int(click_ch_idx),
            seed_t_idx=int(seed_t_idx),
            step_sign=-1,
            direction=direction,
        )
        right = self._kalman_trace_one_side(
            seed_ch_idx=int(click_ch_idx),
            seed_t_idx=int(seed_t_idx),
            step_sign=1,
            direction=direction,
        )

        track = self.project.add_track(direction=direction, source="right_click_kalman")
        self.project.upsert_point(
            track_id=int(track.track_id),
            ch_idx=int(click_ch_idx),
            t_idx=int(seed_t_idx),
            offset_m=float(self.backend.x_axis_m[int(click_ch_idx)]),
            amp=float(seed_amp),
            score=float(seed_score),
            source="right_click_kalman",
        )
        for ch_idx2, t_idx2, amp2, score2 in left + right:
            self.project.upsert_point(
                track_id=int(track.track_id),
                ch_idx=int(ch_idx2),
                t_idx=int(t_idx2),
                offset_m=float(self.backend.x_axis_m[int(ch_idx2)]),
                amp=float(amp2),
                score=float(score2),
                source="right_click_kalman",
            )
        self.project.renumber_tracks()
        self.selected_track_id = int(track.track_id)
        return 1 + len(left) + len(right)

    def _extend_selected_track_graph(self, click_ch_idx: int, click_t_idx: int) -> int:
        if self.project is None or self.selected_track_id is None:
            return 0
        record = self.project.get_track(self.selected_track_id)
        if record is None or not record.points:
            return 0

        points = sorted(record.points, key=lambda p: p.ch_idx)
        anchor = min(
            points,
            key=lambda point: self._track_point_distance_cost(point, click_ch_idx=click_ch_idx, click_t_idx=click_t_idx),
        )
        anchor_ch = int(anchor.ch_idx)
        anchor_t = int(anchor.t_idx)
        if anchor_ch == int(click_ch_idx):
            path_channels = [int(click_ch_idx)]
        else:
            step = 1 if int(click_ch_idx) > anchor_ch else -1
            path_channels = list(range(anchor_ch + step, int(click_ch_idx) + step, step))
        if not path_channels:
            return 0

        cfg = self._graph_config_from_gui()
        nodes = _build_nodes(np.asarray(self.backend.data_all, dtype=np.float32), float(self.backend.fs), cfg)
        delta_ch_total = max(1, abs(int(click_ch_idx) - anchor_ch))
        delta_t_total = int(click_t_idx) - anchor_t
        per_ch_dt = float(delta_t_total) / float(delta_ch_total)
        search_radius = int(
            max(
                120,
                min(
                    1800,
                    round(
                        max(
                            float(self.min_peak_distance_input.text().strip()) * 0.6,
                            abs(per_ch_dt) * 1.2 + 80.0,
                        )
                    ),
                ),
            )
        )

        prev_t_idx = int(anchor_t)
        added_count = 0
        skipped_count = 0
        for path_idx, ch_idx in enumerate(path_channels, start=1):
            frac = float(path_idx) / float(max(1, len(path_channels)))
            guide_t_idx = int(round(float(anchor_t) + frac * float(delta_t_total)))
            cont_t_idx = int(round(float(prev_t_idx) + float(per_ch_dt)))
            snapped = self._snap_channel_time_to_peak(
                nodes=nodes,
                ch_idx=int(ch_idx),
                guide_t_idx=int(guide_t_idx),
                search_radius=int(search_radius),
                continuity_t_idx=int(cont_t_idx),
            )
            if snapped is None:
                skipped_count += 1
                continue
            best_t_idx, best_amp, best_score = snapped
            self.project.upsert_point(
                track_id=int(record.track_id),
                ch_idx=int(ch_idx),
                t_idx=int(best_t_idx),
                offset_m=float(self.backend.x_axis_m[int(ch_idx)]),
                amp=float(best_amp),
                score=float(best_score),
                source="bridge_peak",
            )
            prev_t_idx = int(best_t_idx)
            added_count += 1
        self.status_label.setText(
            f"Bridge result for track #{int(record.track_id)}: added={added_count}, skipped={skipped_count}"
        )
        return int(added_count)

    def _extend_selected_track_kalman(self, click_ch_idx: int, click_t_idx: int) -> int:
        if self.project is None or self.selected_track_id is None:
            return 0
        record = self.project.get_track(self.selected_track_id)
        if record is None or not record.points:
            return 0

        points = sorted(record.points, key=lambda p: p.ch_idx)
        existing_channels = {int(point.ch_idx) for point in points}
        min_point = points[0]
        max_point = points[-1]
        min_ch = int(min_point.ch_idx)
        max_ch = int(max_point.ch_idx)

        if int(click_ch_idx) < min_ch:
            anchor = min_point
            step_sign = -1
            allowed = lambda ch: int(ch) < min_ch
        elif int(click_ch_idx) > max_ch:
            anchor = max_point
            step_sign = 1
            allowed = lambda ch: int(ch) > max_ch
        else:
            self.status_label.setText(
                f"Kalman skipped: click outside selected track #{int(record.track_id)} "
                f"channel span [{min_ch}, {max_ch}] to extend it."
            )
            return 0

        traced = self._kalman_trace_one_side(
            seed_ch_idx=int(anchor.ch_idx),
            seed_t_idx=int(anchor.t_idx),
            step_sign=int(step_sign),
            direction=str(record.direction),
            stop_ch_idx=int(click_ch_idx),
        )
        added_count = 0
        skipped_existing = 0
        for ch_idx2, t_idx2, amp2, score2 in traced:
            if int(ch_idx2) in existing_channels or not allowed(int(ch_idx2)):
                skipped_existing += 1
                continue
            self.project.upsert_point(
                track_id=int(record.track_id),
                ch_idx=int(ch_idx2),
                t_idx=int(t_idx2),
                offset_m=float(self.backend.x_axis_m[int(ch_idx2)]),
                amp=float(amp2),
                score=float(score2),
                source="bridge_kalman_external",
            )
            added_count += 1
        self.status_label.setText(
            f"Kalman external extend for track #{int(record.track_id)}: "
            f"added={added_count}, preserved_existing={len(existing_channels)}, skipped={skipped_existing}"
        )
        return int(added_count)

    def _event_to_global_indices(self, event) -> Optional[tuple[int, int, float, float]]:
        if event is None or event.xdata is None or event.ydata is None:
            return None
        ch_idx = self._nearest_channel_index(float(event.xdata))
        t_local = float(event.ydata)
        t_idx = int(round(float(self.backend.current_start) + t_local * float(self.backend.fs)))
        t_idx = max(0, min(t_idx, int(self.backend.data_all.shape[1] - 1)))
        x_m = float(self.backend.x_axis_m[ch_idx])
        amp = float(self.backend.data_all[ch_idx, t_idx])
        return ch_idx, t_idx, x_m, amp

    def on_plot_click(self, event) -> None:
        if self.project is None:
            return
        if self.backend.data_all.size == 0:
            return
        if getattr(self.toolbar, "mode", ""):
            return
        if getattr(event, "inaxes", None) is not self.ax:
            return
        click_data = self._event_to_global_indices(event)
        if click_data is None:
            return
        ch_idx, t_idx, x_m, amp = click_data
        if self._is_right_click(event):
            if self._click_backend() == "kalman":
                if self.selected_track_id is None:
                    self.status_label.setText("Kalman needs a selected graph label track; select one first.")
                    return
                added_count = self._extend_selected_track_kalman(click_ch_idx=ch_idx, click_t_idx=t_idx)
            else:
                added_count = self._create_track_from_right_click_peak(click_ch_idx=ch_idx, click_t_idx=t_idx)
            self._refresh_track_list()
            self.redraw()
            if self._click_backend() != "kalman":
                self.status_label.setText(
                    f"Right-click {self._click_backend()} trace created: points={added_count}, "
                    f"direction={self.direction_combo.currentData()}"
                )
            return
        start, end = self._current_window_bounds()
        x_tol_m = max(1.0, float(self.backend.dx_m) * 0.75)
        t_tol_samples = max(1, int(round(0.8 * float(self.backend.fs))))
        mode = str(self.edit_mode_combo.currentData())

        if mode == "select":
            nearest = self.project.nearest_point(
                start_sample=start,
                end_sample=end,
                x_m=x_m,
                t_idx=t_idx,
                x_tol_m=x_tol_m,
                t_tol_samples=t_tol_samples,
            )
            if nearest is None:
                return
            track, _point = nearest
            self.selected_track_id = int(track.track_id)
            self._refresh_track_list()
            self.redraw()
            self.status_label.setText(f"Selected label track #{self.selected_track_id}")
            return

        if self.selected_track_id is None:
            QMessageBox.information(self, "Edit Label", "Please select or create a label track first.")
            return

        if mode == "add_move":
            self.project.upsert_point(
                track_id=self.selected_track_id,
                ch_idx=ch_idx,
                t_idx=t_idx,
                offset_m=x_m,
                amp=amp,
                score=1.0,
                source="manual",
            )
            track = self.project.get_track(self.selected_track_id)
            if track is not None:
                track.direction = str(self.direction_combo.currentData())
            self._refresh_track_list()
            self.redraw()
            self.status_label.setText(
                f"Updated label track #{self.selected_track_id}: ch={ch_idx}, t={t_idx}, time={t_idx / self.backend.fs:.3f}s"
            )
            return

        if mode == "delete_point":
            nearest = self.project.nearest_point(
                start_sample=start,
                end_sample=end,
                x_m=x_m,
                t_idx=t_idx,
                x_tol_m=x_tol_m,
                t_tol_samples=t_tol_samples,
                only_track_id=self.selected_track_id,
            )
            if nearest is None:
                self.status_label.setText("No label point close enough to delete")
                return
            track, point = nearest
            self.project.remove_point(track_id=int(track.track_id), ch_idx=int(point.ch_idx))
            if self.selected_track_id is not None and self.project.get_track(self.selected_track_id) is None:
                self.selected_track_id = None
            self._refresh_track_list()
            self.redraw()
            self.status_label.setText(f"Deleted point on label track #{track.track_id}, channel {point.ch_idx}")
            return

        if mode == "extend_graph":
            if self._click_backend() == "kalman":
                added_count = self._extend_selected_track_kalman(click_ch_idx=ch_idx, click_t_idx=t_idx)
            else:
                added_count = self._extend_selected_track_graph(click_ch_idx=ch_idx, click_t_idx=t_idx)
            self._refresh_track_list()
            self.redraw()
            self.status_label.setText(
                f"Extended label track #{self.selected_track_id} with {self._click_backend()}: +{added_count} points"
            )

    def import_data(self) -> None:
        super().import_data()
        self.project = TrackLabelProject(
            source_path=self.backend.files,
            source_kind=str(self.backend.last_import_info.get("input_kind", "unknown")),
            fs_hz=float(self.backend.fs),
            dx_m=float(self.backend.dx_m),
        )
        self.project_path = None
        self.selected_track_id = None
        self._refresh_track_list()
        self.redraw()

    def clear_results(self) -> None:
        super().clear_results()
        self._refresh_track_list()
        self.redraw()

    def on_worker_finished(self, summary: dict) -> None:
        super().on_worker_finished(summary)
        if self.project is None:
            return
        if self._adopt_extract_on_finish:
            self._adopt_extract_on_finish = False
            start, end = self._current_window_bounds()
            tracks = TrackLabelProject.from_backend_tracks(self.backend.tracks, source="auto")
            self.project.replace_tracks_in_window(start_sample=start, end_sample=end, tracks=tracks)
            self._refresh_track_list()
            self.redraw()
            self.status_label.setText(
                f"Graph auto-labeled current window: {len(tracks)} tracks, "
                f"{sum(len(track.points) for track in tracks)} points"
            )
        else:
            self._refresh_track_list()

    def on_worker_failed(self, message: str) -> None:
        self._adopt_extract_on_finish = False
        super().on_worker_failed(message)

    def on_window_slider_changed(self, value: int) -> None:
        super().on_window_slider_changed(value)
        self._refresh_track_list()

    def redraw(self) -> None:
        super().redraw()
        if self.backend.data_view.size == 0 or self.project is None:
            return

        start, end = self._current_window_bounds()
        visible = self.project.visible_tracks(start, end)
        if visible:
            cmap = matplotlib.colormaps.get_cmap("tab20")
            for idx, track in enumerate(visible):
                points = sorted(
                    [point for point in track.points if start <= int(point.t_idx) < end],
                    key=lambda point: int(point.ch_idx),
                )
                if not points:
                    continue
                xs = np.asarray([float(point.offset_m) * 1e-3 for point in points], dtype=np.float64)
                ts = np.asarray([(int(point.t_idx) - start) / float(self.backend.fs) for point in points], dtype=np.float64)
                selected = self.selected_track_id is not None and int(track.track_id) == int(self.selected_track_id)
                color = "#00a651" if selected else cmap(idx % max(1, cmap.N))
                self.ax.plot(xs, ts, color=color, linewidth=2.2 if selected else 1.4, alpha=0.95, zorder=7)
                self.ax.scatter(xs, ts, s=34 if selected else 16, color=[color], marker="o", alpha=0.95, zorder=8)
                mid = len(xs) // 2
                label = f"L#{track.track_id}"
                if np.isfinite(track.mean_speed_kmh):
                    label += f" {track.mean_speed_kmh:.0f}km/h"
                self.ax.text(
                    xs[mid],
                    ts[mid],
                    label,
                    color=color,
                    fontsize=8,
                    ha="left",
                    va="center",
                    alpha=0.95,
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 0.8},
                    zorder=9,
                )

        self.canvas.draw_idle()

    def keyPressEvent(self, event) -> None:
        if event is not None and event.key() == int(Qt.Key.Key_S):
            self.mark_selected_track_saved()
            event.accept()
            return
        super().keyPressEvent(event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-data labeling GUI based on the original graph-search GUI.")
    parser.add_argument(
        "--data-folder",
        type=str,
        default=DEFAULT_DATA_FOLDER,
        help="SAC data folder or real DAS .npy file",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    gui = RealDataLabelGUI(data_folder=args.data_folder)
    gui.resize(2050, 980)
    gui.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
