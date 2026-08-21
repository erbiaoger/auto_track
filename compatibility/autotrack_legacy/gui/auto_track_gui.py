from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QSettings, QSignalBlocker, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
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

from autotrack.core.auto_track_backend import AutoTrackBackend, DEFAULT_DATA_FOLDER


DL_ADVANCED_PARAM_SPECS = [
    {
        "key": "peak_threshold",
        "label": "Peak threshold",
        "type": "float",
        "default": 0.35,
        "tooltip": "PeakSlotNet 点级峰选择阈值。",
    },
    {
        "key": "candidate_source",
        "label": "Candidate source",
        "type": "choice",
        "default": "checkpoint",
        "choices": ["checkpoint", "raw", "prior", "raw_prior_union"],
        "tooltip": "PeakSlot v5 可从 checkpoint meta 继承 raw/prior union 候选来源。",
    },
    {
        "key": "unet_checkpoint",
        "label": "U-Net checkpoint",
        "type": "str",
        "default": "",
        "tooltip": "2 通道 PeakSlot 模型需要的 waveform-line U-Net checkpoint；留空则使用模型 meta。",
    },
    {
        "key": "waveform_task_dir",
        "label": "Waveform task dir",
        "type": "str",
        "default": "/csim2/zhangzhiyu/MyProjects/waveform_line_task",
        "tooltip": "包含 waveform_line_task model/ 和 render.py 的目录。",
    },
    {
        "key": "prior_device",
        "label": "Prior device",
        "type": "str",
        "default": "",
        "tooltip": "U-Net prior 推理设备；留空则跟随 DL device。",
    },
    {
        "key": "prior_batch_size",
        "label": "Prior batch size",
        "type": "int",
        "default": 1,
        "tooltip": "GUI 逐窗口 U-Net prior 推理 batch size。",
    },
    {
        "key": "max_tracks",
        "label": "Max predicted tracks",
        "type": "int",
        "default": 96,
        "tooltip": "每个窗口最多输出多少条深度学习轨迹。",
    },
    {
        "key": "extra_candidate_slots",
        "label": "Extra candidate slots",
        "type": "int",
        "default": 48,
        "tooltip": "额外解码的低 objectness slot 数。",
    },
    {
        "key": "candidate_objectness_floor",
        "label": "Candidate objectness floor",
        "type": "float",
        "default": 0.01,
        "tooltip": "额外候选 slot 的最低 objectness。",
    },
    {
        "key": "decoder_mode",
        "label": "Decoder mode",
        "type": "choice",
        "default": "beam_global",
        "choices": ["beam_global", "beam", "argmax"],
        "tooltip": "PeakSlotNet 解码模式。",
    },
    {
        "key": "viterbi_beam_size",
        "label": "Viterbi beam size",
        "type": "int",
        "default": 8,
        "tooltip": "Viterbi / beam 搜索宽度。",
    },
    {
        "key": "time_prior_weight",
        "label": "Time prior weight",
        "type": "float",
        "default": 2.0,
        "tooltip": "时间先验权重。",
    },
    {
        "key": "global_conflict_penalty",
        "label": "Global conflict penalty",
        "type": "float",
        "default": 2.0,
        "tooltip": "跨 slot 冲突惩罚。",
    },
    {
        "key": "conflict_mode",
        "label": "Conflict mode",
        "type": "choice",
        "default": "soft",
        "choices": ["soft", "hard", "off"],
        "tooltip": "冲突处理模式。",
    },
    {
        "key": "duplicate_overlap_ratio",
        "label": "Duplicate overlap ratio",
        "type": "float",
        "default": 0.70,
        "tooltip": "soft 模式下判定重复轨迹的最小重合比例。",
    },
    {
        "key": "min_unique_support_channels",
        "label": "Min unique support channels",
        "type": "int",
        "default": 4,
        "tooltip": "近车轨迹至少有多少独立峰支撑时保留。",
    },
    {
        "key": "viterbi_topk",
        "label": "Viterbi top-k",
        "type": "int",
        "default": 24,
        "tooltip": "每道保留多少候选峰。",
    },
    {
        "key": "prior_peak_min_height",
        "label": "Prior peak min height",
        "type": "float",
        "default": 0.08,
        "tooltip": "Ponytail prior 候选峰最低概率。",
    },
    {
        "key": "prior_peak_prominence",
        "label": "Prior peak prominence",
        "type": "float",
        "default": 0.02,
        "tooltip": "Ponytail prior 候选峰显著性阈值。",
    },
    {
        "key": "candidate_merge_tolerance_s",
        "label": "Candidate merge tolerance (s)",
        "type": "float",
        "default": 0.20,
        "tooltip": "同一道 raw/prior 候选在多少秒内合并。",
    },
    {
        "key": "prior_score_scale",
        "label": "Prior score scale",
        "type": "float",
        "default": 0.85,
        "tooltip": "raw_prior_union 中 prior 候选分数缩放。",
    },
    {
        "key": "viterbi_candidate_threshold",
        "label": "Viterbi candidate threshold",
        "type": "float",
        "default": 0.01,
        "tooltip": "候选峰最低概率阈值。",
    },
    {
        "key": "viterbi_speed_min_kmh",
        "label": "Viterbi speed min (km/h)",
        "type": "float",
        "default": 60.0,
        "tooltip": "解码允许最小速度。",
    },
    {
        "key": "viterbi_speed_max_kmh",
        "label": "Viterbi speed max (km/h)",
        "type": "float",
        "default": 100.0,
        "tooltip": "解码允许最大速度。",
    },
    {
        "key": "viterbi_max_skip_channels",
        "label": "Viterbi max skip channels",
        "type": "int",
        "default": 4,
        "tooltip": "最多允许跳过多少道。",
    },
    {
        "key": "viterbi_inertia_penalty",
        "label": "Viterbi inertia penalty",
        "type": "float",
        "default": 2.5,
        "tooltip": "速度惯性惩罚。",
    },
    {
        "key": "viterbi_slope_memory",
        "label": "Viterbi slope memory",
        "type": "float",
        "default": 0.75,
        "tooltip": "斜率记忆系数。",
    },
    {
        "key": "boundary_completion_mode",
        "label": "Boundary completion mode",
        "type": "choice",
        "default": "strict",
        "choices": ["off", "repair", "strict"],
        "tooltip": "边界感知轨迹装配模式：repair 优先补全，strict 只保留完整过境轨迹。",
    },
    {
        "key": "boundary_max_gap_channels",
        "label": "Boundary max gap channels",
        "type": "int",
        "default": 12,
        "tooltip": "边界补全和断片连接允许的最大通道缺口。",
    },
    {
        "key": "boundary_fit_residual_s",
        "label": "Boundary fit residual (s)",
        "type": "float",
        "default": 1.0,
        "tooltip": "边界补全直线拟合允许的中位时间残差。",
    },
    {
        "key": "boundary_margin_channels",
        "label": "Boundary margin channels",
        "type": "int",
        "default": 2,
        "tooltip": "距离左右通道边界多少道以内算已有边界支撑。",
    },
    {
        "key": "boundary_margin_seconds",
        "label": "Boundary margin seconds",
        "type": "float",
        "default": 3.0,
        "tooltip": "距离时间窗上下边界多少秒以内算已有边界支撑。",
    },
    {
        "key": "boundary_dt_slack_ratio",
        "label": "Boundary dt slack ratio",
        "type": "float",
        "default": 0.35,
        "tooltip": "边界连接物理速度窗口的时间松弛比例。",
    },
    {
        "key": "boundary_min_seed_channels",
        "label": "Boundary min seed channels",
        "type": "int",
        "default": 4,
        "tooltip": "尝试边界延长时要求的最少已有通道点数。",
    },
    {
        "key": "boundary_projected_completion_enabled",
        "label": "Boundary projected completion",
        "type": "bool",
        "default": True,
        "tooltip": "允许一侧有真实边界支撑且物理投影穿过另一侧边界的长轨迹进入最终结果。",
    },
    {
        "key": "boundary_projected_min_span_channels",
        "label": "Boundary projected min span",
        "type": "int",
        "default": 18,
        "tooltip": "投影补全进入最终结果所需的最小通道跨度，防止短碎片被当作完整车辆。",
    },
]


def _default_dl_advanced_params() -> dict[str, object]:
    return {str(spec["key"]): spec["default"] for spec in DL_ADVANCED_PARAM_SPECS}


def _peak_slot_recommended_advanced_params() -> dict[str, object]:
    return {
        "peak_threshold": 0.35,
        "candidate_source": "checkpoint",
        "unet_checkpoint": "",
        "waveform_task_dir": "/csim2/zhangzhiyu/MyProjects/waveform_line_task",
        "prior_device": "",
        "prior_batch_size": 1,
        "max_tracks": 96,
        "extra_candidate_slots": 48,
        "candidate_objectness_floor": 0.01,
        "decoder_mode": "beam_global",
        "viterbi_beam_size": 8,
        "time_prior_weight": 2.0,
        "global_conflict_penalty": 2.0,
        "conflict_mode": "soft",
        "duplicate_overlap_ratio": 0.70,
        "min_unique_support_channels": 4,
        "viterbi_topk": 24,
        "prior_peak_min_height": 0.08,
        "prior_peak_prominence": 0.02,
        "candidate_merge_tolerance_s": 0.20,
        "prior_score_scale": 0.85,
        "viterbi_candidate_threshold": 0.01,
        "viterbi_speed_min_kmh": 60.0,
        "viterbi_speed_max_kmh": 100.0,
        "viterbi_max_skip_channels": 4,
        "viterbi_inertia_penalty": 2.5,
        "viterbi_slope_memory": 0.75,
        "boundary_completion_mode": "strict",
        "boundary_max_gap_channels": 12,
        "boundary_fit_residual_s": 1.0,
        "boundary_margin_channels": 2,
        "boundary_margin_seconds": 3.0,
        "boundary_dt_slack_ratio": 0.35,
        "boundary_min_seed_channels": 4,
        "boundary_projected_completion_enabled": True,
        "boundary_projected_min_span_channels": 18,
    }


class ExtractWorker(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, backend: AutoTrackBackend, params: dict):
        super().__init__()
        self.backend = backend
        self.params = params

    def run(self) -> None:
        try:
            def _cb(pct: int, msg: str) -> None:
                self.progress.emit(int(pct), str(msg))

            summary = self.backend.run_auto_extract(progress_cb=_cb, **self.params)
            self.finished.emit(summary)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class DeepLearningParamsDialog(QDialog):
    def __init__(self, parent: QWidget | None, current_values: dict[str, object]):
        super().__init__(parent)
        self.setWindowTitle("Deep Learning Parameters")
        self.resize(560, 620)
        self._inputs: dict[str, QWidget] = {}
        self._values: dict[str, object] = dict(current_values)

        layout = QVBoxLayout(self)

        note = QLabel(
            "这里配置 GUI Deep Learning 模式会实际用到的推理参数。"
            " `BATCH_SIZE`、`MAX_SAMPLES`、`MAX_CSV_SAMPLES`、`PLOT_*`、`MATCHER`、"
            "`PLOT_DIRECTION_FILTER`、`PREDICTION_DIRECTION_FILTER` 属于离线数据集预测脚本参数，"
            "当前 GUI 不消费，所以不在这里暴露。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        for spec in DL_ADVANCED_PARAM_SPECS:
            key = str(spec["key"])
            tooltip = str(spec.get("tooltip", ""))
            label = QLabel(str(spec["label"]))
            label.setToolTip(tooltip)
            if str(spec["type"]) == "choice":
                widget = QComboBox()
                for choice in spec.get("choices", []):
                    widget.addItem(str(choice), str(choice))
                idx = widget.findData(str(current_values.get(key, spec["default"])))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            else:
                widget = QLineEdit(str(current_values.get(key, spec["default"])))
            widget.setToolTip(tooltip)
            self._inputs[key] = widget
            form.addRow(label, widget)

        scroll.setWidget(form_host)
        layout.addWidget(scroll, stretch=1)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        reset_btn = button_box.addButton("Reset Defaults", QDialogButtonBox.ButtonRole.ResetRole)
        reset_btn.clicked.connect(self._reset_defaults)
        button_box.accepted.connect(self._accept_with_validation)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _reset_defaults(self) -> None:
        defaults = _default_dl_advanced_params()
        for spec in DL_ADVANCED_PARAM_SPECS:
            key = str(spec["key"])
            widget = self._inputs[key]
            value = defaults[key]
            if isinstance(widget, QComboBox):
                idx = widget.findData(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))

    def _collect_values(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for spec in DL_ADVANCED_PARAM_SPECS:
            key = str(spec["key"])
            widget = self._inputs[key]
            typ = str(spec["type"])
            if isinstance(widget, QComboBox):
                raw = str(widget.currentData())
            elif isinstance(widget, QLineEdit):
                raw = widget.text().strip()
            else:
                raise TypeError(f"Unsupported widget type for {key}")
            if typ == "int":
                values[key] = int(float(raw))
            elif typ == "float":
                values[key] = float(raw)
            elif typ == "choice":
                values[key] = str(raw)
            else:
                values[key] = raw

        if float(values["peak_threshold"]) < 0.0 or float(values["peak_threshold"]) > 1.0:
            raise ValueError("Peak threshold must be in [0, 1].")
        if int(values["prior_batch_size"]) < 1:
            raise ValueError("Prior batch size must be >= 1.")
        if int(values["max_tracks"]) < 1:
            raise ValueError("Max predicted tracks must be >= 1.")
        if int(values["extra_candidate_slots"]) < 0:
            raise ValueError("Extra candidate slots must be >= 0.")
        if float(values["candidate_objectness_floor"]) < 0.0 or float(values["candidate_objectness_floor"]) > 1.0:
            raise ValueError("Candidate objectness floor must be in [0, 1].")
        if int(values["viterbi_beam_size"]) < 1:
            raise ValueError("Viterbi beam size must be >= 1.")
        if float(values["duplicate_overlap_ratio"]) < 0.0 or float(values["duplicate_overlap_ratio"]) > 1.0:
            raise ValueError("Duplicate overlap ratio must be in [0, 1].")
        if int(values["min_unique_support_channels"]) < 0:
            raise ValueError("Min unique support channels must be >= 0.")
        if int(values["viterbi_topk"]) < 1:
            raise ValueError("Viterbi top-k must be >= 1.")
        if float(values["prior_peak_min_height"]) < 0.0:
            raise ValueError("Prior peak min height must be >= 0.")
        if float(values["prior_peak_prominence"]) < 0.0:
            raise ValueError("Prior peak prominence must be >= 0.")
        if float(values["candidate_merge_tolerance_s"]) < 0.0:
            raise ValueError("Candidate merge tolerance must be >= 0.")
        if float(values["prior_score_scale"]) < 0.0:
            raise ValueError("Prior score scale must be >= 0.")
        if float(values["viterbi_candidate_threshold"]) < 0.0 or float(values["viterbi_candidate_threshold"]) > 1.0:
            raise ValueError("Viterbi candidate threshold must be in [0, 1].")
        if float(values["viterbi_speed_min_kmh"]) <= 0.0 or float(values["viterbi_speed_max_kmh"]) <= 0.0:
            raise ValueError("Viterbi speed bounds must be > 0.")
        if float(values["viterbi_speed_min_kmh"]) > float(values["viterbi_speed_max_kmh"]):
            raise ValueError("Viterbi speed min cannot be greater than max.")
        if int(values["viterbi_max_skip_channels"]) < 0:
            raise ValueError("Viterbi max skip channels must be >= 0.")
        return values

    def _accept_with_validation(self) -> None:
        try:
            self._values = self._collect_values()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "DL Parameter Error", str(exc))
            return
        self.accept()

    def values(self) -> dict[str, object]:
        return dict(self._values)


class AutoTrackGUI(QMainWindow):
    def __init__(self, data_folder: str = DEFAULT_DATA_FOLDER):
        super().__init__()
        self.setWindowTitle("Vehicle Trajectory Auto-Extraction System (Peak Map + Dynamic Programming)")
        self.settings = QSettings("BaFang", "AutoTrackGUI")
        self._migrate_persisted_settings()

        self.backend = AutoTrackBackend(data_folder=data_folder)
        self.worker_thread: QThread | None = None
        self.worker: ExtractWorker | None = None
        self._syncing_slider = False
        self._plot_target_points_per_trace = 4000
        self._auto_scroll_paused_by_worker = False
        self._worker_mode: str = "idle"  # idle / manual / auto_follow
        self._auto_extract_pending = False
        self.dl_advanced_params = self._load_dl_advanced_params()
        self.show_peak_slot_diagnostics_check: QCheckBox | None = None

        self.fig = Figure(figsize=(11, 6))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        self.auto_scroll_timer = QTimer(self)
        self.auto_scroll_timer.setInterval(1000)
        self.auto_scroll_timer.timeout.connect(self.on_auto_scroll_tick)

        controls_layout = QVBoxLayout()

        def _apply_tooltip(widget, text: str) -> None:
            widget.setToolTip(text)
            widget.setStatusTip(text)
            widget.setWhatsThis(text)

        def _make_label(text: str, tooltip: str) -> QLabel:
            label = QLabel(text)
            _apply_tooltip(label, tooltip)
            return label

        path_layout = QVBoxLayout()
        data_folder_tip = "输入 SAC 数据目录，或直接输入单个真实 DAS `.npy` 文件、tensor shard `.pt/.pth` 文件。`.npy` 默认按 `[time, channel]` 解释；`.pt` shard 默认载入其中第 1 个样本，导入后可在图上滚轮切换 sample。"
        path_layout.addWidget(_make_label("Data Folder / Input File", data_folder_tip))
        self.path_input = QLineEdit(str(Path(data_folder)))
        self.path_input.setPlaceholderText("Enter SAC folder, .npy file, or tensor shard .pt path")
        _apply_tooltip(self.path_input, data_folder_tip)

        position_xlsx_tip = "可选：开启后按位置表（xlsx）中的里程位置排序通道。默认关闭，不影响原有流程。"
        self.use_position_xlsx_check = QCheckBox("Use Position XLSX Ordering")
        self.use_position_xlsx_check.setChecked(False)
        _apply_tooltip(self.use_position_xlsx_check, position_xlsx_tip)
        self.use_position_xlsx_check.toggled.connect(self.on_position_xlsx_toggled)
        self.position_xlsx_input = QLineEdit("")
        self.position_xlsx_input.setPlaceholderText("Optional: select position table .xlsx")
        _apply_tooltip(self.position_xlsx_input, position_xlsx_tip)
        self.position_xlsx_btn = QPushButton("Browse XLSX...")
        _apply_tooltip(self.position_xlsx_btn, position_xlsx_tip)
        self.position_xlsx_btn.clicked.connect(self.browse_position_xlsx)
        npy_fs_tip = "可选：仅对单个 .npy 导入生效。留空则默认按 1000 Hz 解释。"
        npy_dx_tip = "可选：仅对单个 .npy 导入生效。留空则默认按 100 m 通道间距解释。"
        self.npy_fs_input = QLineEdit("")
        self.npy_fs_input.setPlaceholderText("NPY fs (Hz), optional")
        _apply_tooltip(self.npy_fs_input, npy_fs_tip)
        self.npy_dx_input = QLineEdit("")
        self.npy_dx_input.setPlaceholderText("NPY dx (m), optional")
        _apply_tooltip(self.npy_dx_input, npy_dx_tip)

        path_btn_layout = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        import_btn = QPushButton("Import Data")
        browse_btn.clicked.connect(self.browse_folder)
        import_btn.clicked.connect(self.import_data)
        path_btn_layout.addWidget(browse_btn)
        path_btn_layout.addWidget(import_btn)

        position_xlsx_layout = QHBoxLayout()
        position_xlsx_layout.addWidget(self.position_xlsx_input)
        position_xlsx_layout.addWidget(self.position_xlsx_btn)
        npy_layout = QHBoxLayout()
        npy_layout.addWidget(self.npy_fs_input)
        npy_layout.addWidget(self.npy_dx_input)

        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.use_position_xlsx_check)
        path_layout.addLayout(position_xlsx_layout)
        path_layout.addLayout(npy_layout)
        path_layout.addLayout(path_btn_layout)

        window_lock_layout = QGridLayout()
        window_lock_layout.setHorizontalSpacing(8)
        window_lock_layout.setVerticalSpacing(6)
        self.lock_window_check = QCheckBox("Lock Display Window Length")
        self.lock_window_check.setChecked(True)
        _apply_tooltip(self.lock_window_check, "锁定显示窗口长度。滚动时只移动窗口位置，不自动改变显示时长。")
        self.auto_scroll_check = QCheckBox("Auto-scroll window (1 Hz)")
        self.auto_scroll_check.setChecked(False)
        self.auto_scroll_check.toggled.connect(self.on_auto_scroll_toggled)
        _apply_tooltip(self.auto_scroll_check, "按 1 Hz 自动向后滚动当前显示窗口，适合连续浏览时序数据。")
        self.window_seconds_input = QLineEdit()
        self.window_seconds_input.setPlaceholderText("Window length (s)")
        _apply_tooltip(self.window_seconds_input, "当前显示窗口长度。影响可视化范围，也影响“Extract Current Window”的提取范围。")
        self.apply_window_btn = QPushButton("Apply Window Length")
        self.apply_window_btn.clicked.connect(self.apply_window_length)
        self.lock_window_check.toggled.connect(self.on_lock_window_toggled)
        window_lock_layout.addWidget(self.lock_window_check, 0, 0, 1, 2)
        window_lock_layout.addWidget(self.auto_scroll_check, 1, 0, 1, 2)
        window_lock_layout.addWidget(
            _make_label("Window seconds", "当前显示窗口长度。影响可视化范围，也影响“Extract Current Window”的提取范围。"),
            2,
            0,
        )
        window_lock_layout.addWidget(self.window_seconds_input, 2, 1)
        window_lock_layout.addWidget(self.apply_window_btn, 3, 0, 1, 2)

        params_layout = QGridLayout()
        params_layout.setHorizontalSpacing(8)
        params_layout.setVerticalSpacing(6)

        self.engine_combo = QComboBox()
        self.engine_combo.addItem("CPU Parallel (Recommended)", "cpu_parallel")
        self.engine_combo.addItem("CPU Single-thread (Compatible)", "cpu_single")
        self.engine_combo.addItem("GPU (PyTorch MPS)", "gpu_torch_mps")
        self.engine_combo.addItem("GPU (CuPy / CUDA)", "gpu")
        self.engine_combo.addItem("Deep Learning (PeakSlotNet)", "deep_learning")
        self.engine_combo.setCurrentIndex(2)
        _apply_tooltip(
            self.engine_combo,
            "提取后端选择。Deep Learning 这里固定按 PeakSlotNet 推理；需要选择训练好的 peak_slot PyTorch checkpoint。Apple 芯片可用 GPU (PyTorch MPS)。",
        )

        dl_model_tip = "PeakSlotNet checkpoint（由 train_peak_slot.py 输出）。仅 Deep Learning engine 使用。"
        self.dl_model_path_input = QLineEdit("")
        self.dl_model_path_input.setPlaceholderText("Select peak_slot .pt checkpoint")
        _apply_tooltip(self.dl_model_path_input, dl_model_tip)
        self.dl_model_path_input.textChanged.connect(self._save_dl_model_path)
        self.dl_model_path_btn = QPushButton("Browse Model...")
        _apply_tooltip(self.dl_model_path_btn, dl_model_tip)
        self.dl_model_path_btn.clicked.connect(self.browse_dl_model)
        self.dl_params_btn = QPushButton("DL Params...")
        _apply_tooltip(self.dl_params_btn, "打开 Deep Learning 推理参数面板。")
        self.dl_params_btn.clicked.connect(self.open_dl_params_dialog)
        self.show_peak_slot_diagnostics_check = QCheckBox("PeakSlot diagnostics")
        self.show_peak_slot_diagnostics_check.setChecked(False)
        _apply_tooltip(
            self.show_peak_slot_diagnostics_check,
            "显示 PeakSlotNet 解码诊断叠加：候选峰、被过滤候选轨迹、最终保留轨迹的选峰点。",
        )
        self.show_peak_slot_diagnostics_check.toggled.connect(lambda _checked: self.redraw())

        self.dl_objectness_threshold_input = QLineEdit("0.45")
        _apply_tooltip(self.dl_objectness_threshold_input, "PeakSlotNet objectness 阈值。越高越少误检，但可能漏车。")
        self.dl_visibility_threshold_input = QLineEdit("0.5")
        _apply_tooltip(self.dl_visibility_threshold_input, "每条 PeakSlot 轨迹中通道点是否可见的阈值。")
        self.dl_min_visible_channels_input = QLineEdit("4")
        _apply_tooltip(self.dl_min_visible_channels_input, "一条 PeakSlot 轨迹至少需要多少个可见通道点。")
        self.dl_refine_radius_samples_input = QLineEdit("120")
        _apply_tooltip(self.dl_refine_radius_samples_input, "PeakSlot 解码点附近做局部峰值修正的搜索半径（采样点）。")

        self.direction_combo = QComboBox()
        self.direction_combo.addItem("forward", "forward")
        self.direction_combo.addItem("reverse", "reverse")
        _apply_tooltip(self.direction_combo, "车辆行进方向。它决定只允许连接哪一类斜率方向的轨迹。")

        self.speed_min_input = QLineEdit("60")
        _apply_tooltip(self.speed_min_input, "最小可行速度（km/h）。用于限制建边时间窗；太大容易漏慢车。")
        self.speed_max_input = QLineEdit("120")
        _apply_tooltip(self.speed_max_input, "最大可行速度（km/h）。用于限制建边时间窗；太小容易漏快车。")
        self.template_enhance_check = QCheckBox("Enable Gaussian Template Enhancement")
        self.template_enhance_check.setChecked(False)
        _apply_tooltip(self.template_enhance_check, "先做多尺度高斯模板增强再找峰。低信噪比下更稳，但会让响应更平滑。")
        self.prominence_input = QLineEdit("0.4")
        _apply_tooltip(self.prominence_input, "峰值显著性阈值。越大候选峰越少、越干净；越小越容易出现噪声峰。")
        self.min_peak_distance_input = QLineEdit("500")
        _apply_tooltip(self.min_peak_distance_input, "同一道上相邻候选峰的最小间隔（采样点）。越大越不容易一辆车出多个近邻峰。")
        self.min_track_channels_input = QLineEdit("12")
        _apply_tooltip(self.min_track_channels_input, "普通区域轨迹至少包含多少个路径点。越大越严格，碎片更少，但短轨迹更容易被过滤。")
        self.edge_min_track_channels_input = QLineEdit("4")
        _apply_tooltip(self.edge_min_track_channels_input, "窗口边界附近放宽后的最小轨迹长度。适合保留刚进入或刚离开窗口的短轨迹。")
        self.edge_time_margin_seconds_input = QLineEdit("15")
        _apply_tooltip(self.edge_time_margin_seconds_input, "距离窗口起点/终点多近才算“边界区域”（秒）。边界区会使用更宽松的轨迹门槛。")
        self.edge_min_score_scale_input = QLineEdit("0.2")
        _apply_tooltip(self.edge_min_score_scale_input, "边界区域分数缩放系数。边界最小分数 = min_track_score × 本参数。越小越容易保留边界短轨迹。")
        self.tile_seconds_input = QLineEdit("120")
        _apply_tooltip(self.tile_seconds_input, "整小时模式下的分块长度（秒）。越大越稳，越小越快。当前窗口提取基本不受它影响。")
        self.overlap_seconds_input = QLineEdit("20")
        _apply_tooltip(self.overlap_seconds_input, "整小时模式相邻分块的重叠时间（秒）。太小会断轨，太大则重复轨迹更多。")
        self.nms_time_radius_input = QLineEdit("0.18")
        _apply_tooltip(self.nms_time_radius_input, "每提取一条轨迹后，在其时间邻域内抑制候选点的半径（秒）。太小易重复提取，太大易压掉邻近车辆。")

        row = 0
        params_layout.addWidget(
            _make_label(
                "Extractor engine",
                "提取后端选择。Apple 芯片优先用 GPU (PyTorch MPS)，CUDA 机器可用 CuPy，CPU Parallel 兼容性最好。",
            ),
            row,
            0,
        )
        params_layout.addWidget(self.engine_combo, row, 1)
        row += 1
        dl_model_layout = QHBoxLayout()
        dl_model_layout.addWidget(self.dl_model_path_input)
        dl_model_layout.addWidget(self.dl_model_path_btn)
        dl_model_layout.addWidget(self.dl_params_btn)
        params_layout.addWidget(_make_label("DL model", dl_model_tip), row, 0)
        params_layout.addLayout(dl_model_layout, row, 1)
        row += 1
        params_layout.addWidget(QLabel(""), row, 0)
        params_layout.addWidget(self.show_peak_slot_diagnostics_check, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("DL objectness", "Deep Learning query 置信度阈值。越高越少误检，但可能漏车。"),
            row,
            0,
        )
        params_layout.addWidget(self.dl_objectness_threshold_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("DL visibility", "每条轨迹中通道点是否可见的阈值。"),
            row,
            0,
        )
        params_layout.addWidget(self.dl_visibility_threshold_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("DL min visible", "一条深度学习轨迹至少需要多少个可见通道点。"),
            row,
            0,
        )
        params_layout.addWidget(self.dl_min_visible_channels_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("DL refine radius", "深度学习预测点附近做局部峰值修正的搜索半径（采样点）。"),
            row,
            0,
        )
        params_layout.addWidget(self.dl_refine_radius_samples_input, row, 1)
        row += 1
        params_layout.addWidget(_make_label("Direction", "车辆行进方向。它决定只允许连接哪一类斜率方向的轨迹。"), row, 0)
        params_layout.addWidget(self.direction_combo, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("Speed min (km/h)", "最小可行速度（km/h）。用于限制建边时间窗；太大容易漏慢车。"),
            row,
            0,
        )
        params_layout.addWidget(self.speed_min_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("Speed max (km/h)", "最大可行速度（km/h）。用于限制建边时间窗；太小容易漏快车。"),
            row,
            0,
        )
        params_layout.addWidget(self.speed_max_input, row, 1)
        row += 1
        params_layout.addWidget(self.template_enhance_check, row, 0, 1, 2)
        row += 1
        params_layout.addWidget(
            _make_label("Prominence", "峰值显著性阈值。越大候选峰越少、越干净；越小越容易出现噪声峰。"),
            row,
            0,
        )
        params_layout.addWidget(self.prominence_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("Min peak distance", "同一道上相邻候选峰的最小间隔（采样点）。越大越不容易一辆车出多个近邻峰。"),
            row,
            0,
        )
        params_layout.addWidget(self.min_peak_distance_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("Min track channels", "普通区域轨迹至少包含多少个路径点。越大越严格，碎片更少，但短轨迹更容易被过滤。"),
            row,
            0,
        )
        params_layout.addWidget(self.min_track_channels_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("Edge min track channels", "窗口边界附近放宽后的最小轨迹长度。适合保留刚进入或刚离开窗口的短轨迹。"),
            row,
            0,
        )
        params_layout.addWidget(self.edge_min_track_channels_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("Edge margin seconds", "距离窗口起点/终点多近才算“边界区域”（秒）。边界区会使用更宽松的轨迹门槛。"),
            row,
            0,
        )
        params_layout.addWidget(self.edge_time_margin_seconds_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("Edge min score scale", "边界区域分数缩放系数。边界最小分数 = min_track_score × 本参数。越小越容易保留边界短轨迹。"),
            row,
            0,
        )
        params_layout.addWidget(self.edge_min_score_scale_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("Tile seconds", "整小时模式下的分块长度（秒）。越大越稳，越小越快。当前窗口提取基本不受它影响。"),
            row,
            0,
        )
        params_layout.addWidget(self.tile_seconds_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("Overlap seconds", "整小时模式相邻分块的重叠时间（秒）。太小会断轨，太大则重复轨迹更多。"),
            row,
            0,
        )
        params_layout.addWidget(self.overlap_seconds_input, row, 1)
        row += 1
        params_layout.addWidget(
            _make_label("NMS time radius (s)", "每提取一条轨迹后，在其时间邻域内抑制候选点的半径（秒）。太小易重复提取，太大易压掉邻近车辆。"),
            row,
            0,
        )
        params_layout.addWidget(self.nms_time_radius_input, row, 1)
        row += 1

        button_layout = QHBoxLayout()
        self.run_btn = QPushButton("Auto Extract (Full Hour)")
        self.run_window_btn = QPushButton("Extract Current Window")
        self.clear_btn = QPushButton("Clear Results")
        self.export_btn = QPushButton("Export CSV")
        self.run_btn.clicked.connect(lambda: self.start_extract(current_window_only=False))
        self.run_window_btn.clicked.connect(lambda: self.start_extract(current_window_only=True))
        self.clear_btn.clicked.connect(self.clear_results)
        self.export_btn.clicked.connect(self.export_csv)
        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(self.run_window_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.export_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)

        controls_layout.addLayout(path_layout)
        controls_layout.addLayout(window_lock_layout)
        controls_layout.addWidget(QLabel("Auto-Extraction Parameters"))
        controls_layout.addLayout(params_layout)
        controls_layout.addLayout(button_layout)
        controls_layout.addWidget(self.progress)
        controls_layout.addWidget(self.status_label)
        controls_layout.addStretch(1)

        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setMinimumWidth(340)
        controls_widget.setMaximumWidth(460)

        self.window_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_slider.setMinimum(0)
        self.window_slider.setMaximum(0)
        self.window_slider.setSingleStep(1)
        self.window_slider.setPageStep(1)
        self.window_slider.setValue(0)
        self.window_slider.valueChanged.connect(self.on_window_slider_changed)
        self.window_slider_label = QLabel("Window Start: 0.0 s")

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

        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        if self.backend.init_error:
            self.status_label.setText(f"Startup note: {self.backend.init_error}")
        self._connect_persisted_inputs()
        self._restore_persisted_inputs()
        self._load_startup_data()
        self.on_position_xlsx_toggled(self.use_position_xlsx_check.isChecked())
        self._update_window_slider()
        self.redraw()

    def _migrate_persisted_settings(self) -> None:
        version = int(self.settings.value("settings_schema_version", 0) or 0)
        if version < 2:
            old_obj = str(self.settings.value("dl_objectness_threshold_input", "", type=str) or "").strip()
            old_min_vis = str(self.settings.value("dl_min_visible_channels_input", "", type=str) or "").strip()
            if old_obj in {"", "0.35"}:
                self.settings.setValue("dl_objectness_threshold_input", "0.45")
            if old_min_vis in {"", "2"}:
                self.settings.setValue("dl_min_visible_channels_input", "4")
            self.settings.setValue("settings_schema_version", 2)
        if version < 3:
            old_obj = str(self.settings.value("dl_objectness_threshold_input", "", type=str) or "").strip()
            old_min_vis = str(self.settings.value("dl_min_visible_channels_input", "", type=str) or "").strip()
            if old_obj in {"", "0.35", "0.40"}:
                self.settings.setValue("dl_objectness_threshold_input", "0.45")
            if old_min_vis in {"", "1", "2", "3"}:
                self.settings.setValue("dl_min_visible_channels_input", "4")

            raw_adv = str(self.settings.value("dl_advanced_params_json", "", type=str) or "").strip()
            if raw_adv:
                try:
                    adv = json.loads(raw_adv)
                except Exception:  # noqa: BLE001
                    adv = {}
            else:
                adv = {}
            if not isinstance(adv, dict):
                adv = {}
            merged = dict(_default_dl_advanced_params())
            merged.update(adv)
            recommended = _peak_slot_recommended_advanced_params()
            legacy_like = (
                int(merged.get("extra_candidate_slots", 8)) >= 16
                or float(merged.get("candidate_objectness_floor", 0.2)) < 0.1
                or float(merged.get("duplicate_overlap_ratio", 0.7)) >= 0.75
                or int(merged.get("min_unique_support_channels", 4)) <= 3
            )
            if legacy_like or not raw_adv:
                merged.update(recommended)
                self.settings.setValue("dl_advanced_params_json", json.dumps(merged, ensure_ascii=True))
            self.settings.setValue("settings_schema_version", 3)

    def _persisted_line_edits(self) -> dict[str, QLineEdit]:
        return {
            "path_input": self.path_input,
            "position_xlsx_input": self.position_xlsx_input,
            "npy_fs_input": self.npy_fs_input,
            "npy_dx_input": self.npy_dx_input,
            "window_seconds_input": self.window_seconds_input,
            "dl_model_path_input": self.dl_model_path_input,
            "dl_objectness_threshold_input": self.dl_objectness_threshold_input,
            "dl_visibility_threshold_input": self.dl_visibility_threshold_input,
            "dl_min_visible_channels_input": self.dl_min_visible_channels_input,
            "dl_refine_radius_samples_input": self.dl_refine_radius_samples_input,
            "speed_min_input": self.speed_min_input,
            "speed_max_input": self.speed_max_input,
            "prominence_input": self.prominence_input,
            "min_peak_distance_input": self.min_peak_distance_input,
            "min_track_channels_input": self.min_track_channels_input,
            "edge_min_track_channels_input": self.edge_min_track_channels_input,
            "edge_time_margin_seconds_input": self.edge_time_margin_seconds_input,
            "edge_min_score_scale_input": self.edge_min_score_scale_input,
            "tile_seconds_input": self.tile_seconds_input,
            "overlap_seconds_input": self.overlap_seconds_input,
            "nms_time_radius_input": self.nms_time_radius_input,
        }

    def _persisted_checkboxes(self) -> dict[str, QCheckBox]:
        return {
            "use_position_xlsx_check": self.use_position_xlsx_check,
            "lock_window_check": self.lock_window_check,
            "auto_scroll_check": self.auto_scroll_check,
            "template_enhance_check": self.template_enhance_check,
            "show_peak_slot_diagnostics_check": self.show_peak_slot_diagnostics_check,
        }

    def _persisted_combos(self) -> dict[str, QComboBox]:
        return {
            "engine_combo": self.engine_combo,
            "direction_combo": self.direction_combo,
        }

    def _connect_persisted_inputs(self) -> None:
        for widget in self._persisted_line_edits().values():
            widget.textChanged.connect(self._save_persisted_inputs)
        for widget in self._persisted_checkboxes().values():
            widget.toggled.connect(self._save_persisted_inputs)
        for widget in self._persisted_combos().values():
            widget.currentIndexChanged.connect(self._save_persisted_inputs)

    def _restore_persisted_inputs(self) -> None:
        for key, widget in self._persisted_line_edits().items():
            if key == "dl_model_path_input":
                continue
            stored = str(self.settings.value(key, "", type=str) or "")
            if not stored:
                continue
            with QSignalBlocker(widget):
                widget.setText(stored)

        for key, widget in self._persisted_checkboxes().items():
            stored = self.settings.value(key, None)
            if stored is None:
                continue
            with QSignalBlocker(widget):
                widget.setChecked(str(stored).lower() in {"1", "true", "yes"})

        for key, widget in self._persisted_combos().items():
            stored = str(self.settings.value(key, "", type=str) or "")
            if not stored:
                continue
            idx = widget.findData(stored)
            if idx >= 0:
                with QSignalBlocker(widget):
                    widget.setCurrentIndex(idx)

        last_dl_model_path = str(self.settings.value("last_dl_model_path", "", type=str) or "").strip()
        if last_dl_model_path:
            path = Path(last_dl_model_path).expanduser()
            if path.is_file():
                with QSignalBlocker(self.dl_model_path_input):
                    self.dl_model_path_input.setText(str(path))
            else:
                self.settings.remove("last_dl_model_path")
        self._apply_restored_window_seconds()

    def _apply_restored_window_seconds(self) -> None:
        if self.backend.fs <= 0:
            return
        text = self.window_seconds_input.text().strip()
        if not text:
            self._sync_window_seconds_input(force=True)
            return
        try:
            window_seconds = float(text)
            if window_seconds <= 0:
                raise ValueError
            new_size = int(round(window_seconds * self.backend.fs))
            max_samples = int(self.backend.data_all.shape[1]) if self.backend.data_all.size > 0 else new_size
            new_size = max(1000, min(new_size, max_samples))
            self.backend.window_size = new_size
            if self.backend.data_all.size > 0:
                self.backend.current_start = min(
                    int(self.backend.current_start),
                    max(0, max_samples - new_size),
                )
                self.backend.update_view_window()
            with QSignalBlocker(self.window_seconds_input):
                self.window_seconds_input.setText(f"{float(self.backend.window_size) / float(self.backend.fs):.2f}")
        except Exception:  # noqa: BLE001
            self._sync_window_seconds_input(force=True)

    def _load_startup_data(self) -> None:
        startup_path = self.path_input.text().strip()
        if not startup_path:
            return
        use_position_xlsx = bool(self.use_position_xlsx_check.isChecked())
        position_xlsx_path = self.position_xlsx_input.text().strip()
        npy_fs_hz = self._parse_optional_positive_float(self.npy_fs_input)
        npy_dx_m = self._parse_optional_positive_float(self.npy_dx_input)
        try:
            self.backend.load_data_folder(
                startup_path,
                reset_results=True,
                use_position_xlsx=use_position_xlsx,
                position_xlsx_path=position_xlsx_path if use_position_xlsx and position_xlsx_path else None,
                npy_fs_hz=npy_fs_hz,
                npy_dx_m=npy_dx_m,
            )
        except Exception as exc:  # noqa: BLE001
            self.status_label.setText(f"Startup note: failed to load saved input: {exc}")
            return
        self.path_input.setText(self.backend.files)
        self.progress.setValue(0)
        self._set_import_status(prefix="Startup loaded")
        self._apply_restored_window_seconds()

    def _load_dl_advanced_params(self) -> dict[str, object]:
        defaults = _default_dl_advanced_params()
        raw = str(self.settings.value("dl_advanced_params_json", "", type=str) or "").strip()
        if not raw:
            return defaults
        try:
            loaded = json.loads(raw)
        except Exception:  # noqa: BLE001
            return defaults
        if not isinstance(loaded, dict):
            return defaults
        merged = dict(defaults)
        for spec in DL_ADVANCED_PARAM_SPECS:
            key = str(spec["key"])
            if key in loaded:
                merged[key] = loaded[key]
        return merged

    def _save_dl_model_path(self, text: str) -> None:
        value = str(text).strip()
        if value:
            self.settings.setValue("last_dl_model_path", value)
        else:
            self.settings.remove("last_dl_model_path")

    def _save_persisted_inputs(self, *_args) -> None:
        for key, widget in self._persisted_line_edits().items():
            if key == "dl_model_path_input":
                continue
            self.settings.setValue(key, widget.text())
        for key, widget in self._persisted_checkboxes().items():
            self.settings.setValue(key, bool(widget.isChecked()))
        for key, widget in self._persisted_combos().items():
            self.settings.setValue(key, str(widget.currentData()))

    def _save_dl_advanced_params(self) -> None:
        self.settings.setValue("dl_advanced_params_json", json.dumps(self.dl_advanced_params, ensure_ascii=True))

    def open_dl_params_dialog(self) -> None:
        dialog = DeepLearningParamsDialog(self, self.dl_advanced_params)
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return
        self.dl_advanced_params = dialog.values()
        self._save_dl_advanced_params()
        self.status_label.setText("Deep Learning advanced parameters updated")

    def browse_folder(self) -> None:
        start_dir = self.path_input.text().strip() or str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input File",
            start_dir,
            "Supported Files (*.npy *.pt *.pth);;NumPy Array (*.npy);;Tensor Shard (*.pt *.pth);;All Files (*)",
        )
        if selected:
            self.path_input.setText(selected)
            return
        selected = QFileDialog.getExistingDirectory(self, "Select SAC Data Folder", start_dir)
        if selected:
            self.path_input.setText(selected)

    def browse_position_xlsx(self) -> None:
        start_dir = self.position_xlsx_input.text().strip()
        if start_dir:
            start_dir = str(Path(start_dir).expanduser().parent)
        else:
            start_dir = str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Position XLSX",
            start_dir,
            "Excel Files (*.xlsx *.xlsm *.xls);;All Files (*)",
        )
        if selected:
            self.position_xlsx_input.setText(selected)

    def browse_dl_model(self) -> None:
        start_dir = self.dl_model_path_input.text().strip()
        if start_dir:
            start_dir = str(Path(start_dir).expanduser().parent)
        else:
            start_dir = str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Deep Learning Model Checkpoint",
            start_dir,
            "PyTorch Checkpoints (*.pt *.pth);;All Files (*)",
        )
        if selected:
            self.dl_model_path_input.setText(selected)

    def on_position_xlsx_toggled(self, checked: bool) -> None:
        allow_edit = bool(checked) and (self.worker_thread is None)
        self.position_xlsx_input.setEnabled(allow_edit)
        self.position_xlsx_btn.setEnabled(allow_edit)

    def import_data(self) -> None:
        folder = self.path_input.text().strip()
        if not folder:
            QMessageBox.warning(self, "Import Failed", "Please enter a SAC folder, .npy file, or tensor shard .pt path.")
            return
        use_position_xlsx = bool(self.use_position_xlsx_check.isChecked())
        position_xlsx_path = self.position_xlsx_input.text().strip()
        npy_fs_hz = self._parse_optional_positive_float(self.npy_fs_input)
        npy_dx_m = self._parse_optional_positive_float(self.npy_dx_input)
        if use_position_xlsx and (not position_xlsx_path):
            QMessageBox.warning(self, "Import Failed", "Please select a position XLSX file first.")
            return
        try:
            self.auto_scroll_timer.stop()
            self._auto_scroll_paused_by_worker = False
            self.backend.load_data_folder(
                folder,
                reset_results=True,
                use_position_xlsx=use_position_xlsx,
                position_xlsx_path=position_xlsx_path if use_position_xlsx else None,
                npy_fs_hz=npy_fs_hz,
                npy_dx_m=npy_dx_m,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Import Failed", str(exc))
            self.status_label.setText(f"Import failed: {exc}")
            return
        self.path_input.setText(self.backend.files)
        self.progress.setValue(0)
        self._set_import_status(prefix="Imported")
        self._sync_window_seconds_input(force=True)
        self._update_window_slider()
        self.redraw()
        if self.auto_scroll_check.isChecked() and self.worker_thread is None:
            self.auto_scroll_timer.start()

    @staticmethod
    def _parse_optional_positive_float(widget: QLineEdit) -> Optional[float]:
        text = widget.text().strip()
        if not text:
            return None
        value = float(text)
        if value <= 0:
            raise ValueError
        return float(value)

    def _set_import_status(self, *, prefix: str) -> None:
        import_info = dict(getattr(self.backend, "last_import_info", {}))
        if str(import_info.get("input_kind")) == "npy":
            self.status_label.setText(
                f"{prefix} NPY: "
                f"{self.backend.files}\n"
                f"Layout={import_info.get('array_layout')}, "
                f"channels={import_info.get('channels')}, "
                f"samples={import_info.get('samples')}, "
                f"fs={float(import_info.get('fs_hz', 0.0)):.3f} Hz, "
                f"dx={float(import_info.get('dx_m', 0.0)):.3f} m"
            )
            return
        if str(import_info.get("input_kind")) == "tensor_shard":
            sample_index = int(import_info.get("sample_index", 0)) + 1
            sample_count = int(import_info.get("sample_count", 0))
            self.status_label.setText(
                f"{prefix} tensor shard sample: "
                f"{self.backend.files}\n"
                f"sample={sample_index}/{sample_count}, "
                f"dataset={import_info.get('dataset_format') or 'unknown'}, "
                f"channels={import_info.get('channels')}, "
                f"samples={import_info.get('samples')}, "
                f"fs={float(import_info.get('fs_hz', 0.0)):.3f} Hz"
            )
            return
        if bool(import_info.get("position_sort_enabled")):
            matched = int(import_info.get("matched_channels", 0))
            total = int(import_info.get("total_channels", 0))
            unmatched = int(import_info.get("unmatched_channels", max(0, total - matched)))
            suffix = f", unmatched={unmatched} (appended at end)" if unmatched > 0 else ", all matched"
            self.status_label.setText(
                f"{prefix}: {self.backend.files}\nPosition XLSX order ON: matched={matched}/{total}{suffix}"
            )
            return
        self.status_label.setText(f"{prefix}: {self.backend.files}")

    def _switch_tensor_shard_sample(self, delta: int) -> bool:
        if self.worker_thread is not None:
            return False
        import_info = dict(getattr(self.backend, "last_import_info", {}))
        if str(import_info.get("input_kind")) != "tensor_shard":
            return False
        sample_count = int(import_info.get("sample_count", 0))
        if sample_count <= 1:
            return False
        current = int(import_info.get("sample_index", 0))
        target = max(0, min(sample_count - 1, current + int(delta)))
        if target == current:
            return False
        self.auto_scroll_timer.stop()
        self._auto_scroll_paused_by_worker = False
        self.backend.load_tensor_shard_sample(target, reset_results=True)
        self.progress.setValue(0)
        self._sync_window_seconds_input(force=True)
        self._update_window_slider()
        self.redraw()
        self._set_import_status(prefix="Switched")
        return True

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.run_window_btn.setEnabled(not running)
        self.clear_btn.setEnabled(not running)
        self.export_btn.setEnabled(not running)
        self.apply_window_btn.setEnabled(not running)
        self.engine_combo.setEnabled(not running)
        self.dl_model_path_input.setEnabled(not running)
        self.dl_model_path_btn.setEnabled(not running)
        self.dl_params_btn.setEnabled(not running)
        self.dl_objectness_threshold_input.setEnabled(not running)
        self.dl_visibility_threshold_input.setEnabled(not running)
        self.dl_min_visible_channels_input.setEnabled(not running)
        self.dl_refine_radius_samples_input.setEnabled(not running)
        self.template_enhance_check.setEnabled(not running)
        if self.show_peak_slot_diagnostics_check is not None:
            self.show_peak_slot_diagnostics_check.setEnabled(not running)
        self.lock_window_check.setEnabled(not running)
        self.window_seconds_input.setEnabled(not running)
        self.auto_scroll_check.setEnabled(not running)
        self.use_position_xlsx_check.setEnabled(not running)
        self.position_xlsx_input.setEnabled((not running) and self.use_position_xlsx_check.isChecked())
        self.position_xlsx_btn.setEnabled((not running) and self.use_position_xlsx_check.isChecked())
        if running and self.auto_scroll_timer.isActive():
            self.auto_scroll_timer.stop()
            self._auto_scroll_paused_by_worker = True
        elif (not running) and self._auto_scroll_paused_by_worker and self.auto_scroll_check.isChecked():
            self.auto_scroll_timer.start()
            self._auto_scroll_paused_by_worker = False
        if running:
            self.window_slider.setEnabled(False)
        else:
            self._update_window_slider()

    @staticmethod
    def _engine_text(engine_key: str) -> str:
        mapping = {
            "cpu_parallel": "CPU Parallel",
            "cpu_single": "CPU Single-thread",
            "gpu_torch_mps": "GPU(PyTorch MPS)",
            "gpu": "GPU",
            "deep_learning": "Deep Learning(PeakSlotNet)",
        }
        return mapping.get(str(engine_key), str(engine_key))

    def _sync_window_seconds_input(self, force: bool = False) -> None:
        if self.backend.fs <= 0:
            return
        if (not force) and self.window_seconds_input.hasFocus():
            return
        sec = float(self.backend.window_size) / float(self.backend.fs)
        self.window_seconds_input.setText(f"{sec:.2f}")

    def on_lock_window_toggled(self, checked: bool) -> None:
        self.status_label.setText("Window length is locked" if checked else "Window length can be resized (Ctrl + mouse wheel)")

    def on_auto_scroll_toggled(self, checked: bool) -> None:
        if checked:
            if self.backend.data_all.size == 0:
                QMessageBox.warning(self, "Notice", "Please import data first.")
                self.auto_scroll_check.blockSignals(True)
                self.auto_scroll_check.setChecked(False)
                self.auto_scroll_check.blockSignals(False)
                return
            if self.worker_thread is not None:
                self.auto_scroll_check.blockSignals(True)
                self.auto_scroll_check.setChecked(False)
                self.auto_scroll_check.blockSignals(False)
                return
            self.auto_scroll_timer.start()
            self.status_label.setText("Auto-scroll enabled (1 Hz), auto-follow detection enabled")
            self.start_extract(current_window_only=True, auto_follow=True)
            return
        self.auto_scroll_timer.stop()
        self._auto_scroll_paused_by_worker = False
        self.status_label.setText("Auto-scroll disabled")

    def on_auto_scroll_tick(self) -> None:
        if self.backend.data_all.size == 0:
            self.auto_scroll_timer.stop()
            return
        max_start = max(0, int(self.backend.data_all.shape[1] - self.backend.window_size))
        if max_start <= 0:
            return
        step_samples = max(1, int(round(float(self.backend.fs))))
        cur = int(self.backend.current_start)
        new_start = min(max_start, cur + step_samples)
        if new_start <= cur:
            self.auto_scroll_timer.stop()
            self.auto_scroll_check.blockSignals(True)
            self.auto_scroll_check.setChecked(False)
            self.auto_scroll_check.blockSignals(False)
            self.status_label.setText("Auto-scroll reached the data end")
            return
        self.backend.current_start = new_start
        if self.backend.update_view_window():
            self.redraw()
        else:
            self._update_window_slider()
        # Keep detection synchronized with scrolling: auto-extract current window.
        self.start_extract(current_window_only=True, auto_follow=True)

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
            new_size = max(1000, min(new_size, max_samples))
            self.backend.window_size = new_size
            self.backend.current_start = min(
                int(self.backend.current_start),
                max(0, max_samples - new_size),
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
        step = int(max(1, min(self.backend.scroll_step_samples, self.backend.window_size // 5)))
        page = int(max(1, self.backend.window_size // 2))

        self._syncing_slider = True
        self.window_slider.setRange(0, max_start)
        self.window_slider.setSingleStep(step)
        self.window_slider.setPageStep(page)
        self.window_slider.setValue(cur)
        self._syncing_slider = False
        self.window_slider.setEnabled(self.worker_thread is None and max_start > 0)

        t0 = cur / self.backend.fs
        t1 = (cur + self.backend.window_size) / self.backend.fs
        self.window_slider_label.setText(
            f"Window Start: {t0:.1f} s   End: {t1:.1f} s"
        )

    def on_window_slider_changed(self, value: int) -> None:
        if self._syncing_slider:
            return
        if self.backend.data_all.size == 0:
            return
        max_start = max(0, int(self.backend.data_all.shape[1] - self.backend.window_size))
        new_start = int(max(0, min(int(value), max_start)))
        if new_start == int(self.backend.current_start):
            self._update_window_slider()
            return
        self.backend.current_start = new_start
        changed = self.backend.update_view_window()
        if changed:
            self.redraw()
        else:
            self._update_window_slider()

    def _build_params(self, current_window_only: bool = False) -> dict:
        engine = str(self.engine_combo.currentData())
        direction = str(self.direction_combo.currentData())
        speed_min_kmh = float(self.speed_min_input.text().strip())
        speed_max_kmh = float(self.speed_max_input.text().strip())
        prominence = float(self.prominence_input.text().strip())
        min_peak_distance = int(float(self.min_peak_distance_input.text().strip()))
        min_track_channels = int(float(self.min_track_channels_input.text().strip()))
        edge_min_track_channels = int(float(self.edge_min_track_channels_input.text().strip()))
        edge_time_margin_seconds = float(self.edge_time_margin_seconds_input.text().strip())
        edge_min_score_scale = float(self.edge_min_score_scale_input.text().strip())
        tile_seconds = float(self.tile_seconds_input.text().strip())
        overlap_seconds = float(self.overlap_seconds_input.text().strip())
        nms_time_radius = float(self.nms_time_radius_input.text().strip())
        dl_model_path = self.dl_model_path_input.text().strip()
        dl_objectness_threshold = float(self.dl_objectness_threshold_input.text().strip())
        dl_visibility_threshold = float(self.dl_visibility_threshold_input.text().strip())
        dl_min_visible_channels = int(float(self.dl_min_visible_channels_input.text().strip()))
        dl_refine_radius_samples = int(float(self.dl_refine_radius_samples_input.text().strip()))
        enable_template_enhancement = bool(self.template_enhance_check.isChecked())
        if speed_min_kmh > speed_max_kmh:
            raise ValueError("speed_min(km/h) cannot be greater than speed_max(km/h)")
        if engine == "deep_learning" and not dl_model_path:
            raise ValueError("Please select a Deep Learning model checkpoint first.")
        if not (0.0 <= dl_objectness_threshold <= 1.0):
            raise ValueError("DL objectness threshold must be in [0, 1]")
        if not (0.0 <= dl_visibility_threshold <= 1.0):
            raise ValueError("DL visibility threshold must be in [0, 1]")
        if dl_min_visible_channels < 1:
            raise ValueError("DL min visible channels must be >= 1")
        if dl_refine_radius_samples < 0:
            raise ValueError("DL refine radius must be >= 0")
        if edge_min_track_channels < 2:
            raise ValueError("edge_min_track_channels must be >= 2")
        if edge_time_margin_seconds < 0:
            raise ValueError("edge_time_margin_seconds cannot be negative")
        if edge_min_score_scale < 0:
            raise ValueError("edge_min_score_scale cannot be negative")
        if (not current_window_only) and overlap_seconds >= tile_seconds:
            raise ValueError("overlap_seconds must be smaller than tile_seconds")
        return {
            "engine": engine,
            "direction": direction,
            "speed_min_kmh": speed_min_kmh,
            "speed_max_kmh": speed_max_kmh,
            "prominence": prominence,
            "min_peak_distance": min_peak_distance,
            "min_track_channels": min_track_channels,
            "edge_min_track_channels": edge_min_track_channels,
            "edge_time_margin_seconds": edge_time_margin_seconds,
            "edge_min_score_scale": edge_min_score_scale,
            "tile_seconds": tile_seconds,
            "overlap_seconds": overlap_seconds,
            "nms_time_radius": nms_time_radius,
            "enable_template_enhancement": enable_template_enhancement,
            "dl_model_path": dl_model_path,
            "dl_model_family": "peak_slot",
            "dl_objectness_threshold": dl_objectness_threshold,
            "dl_visibility_threshold": dl_visibility_threshold,
            "dl_min_visible_channels": dl_min_visible_channels,
            "dl_refine_radius_samples": dl_refine_radius_samples,
            "dl_extra_config": dict(self.dl_advanced_params),
            "current_window_only": bool(current_window_only),
        }

    def start_extract(self, current_window_only: bool = False, auto_follow: bool = False) -> None:
        if self.worker_thread is not None:
            if auto_follow:
                self._auto_extract_pending = True
            return
        try:
            params = self._build_params(current_window_only=current_window_only)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Parameter Error", str(exc))
            return

        engine_text = self._engine_text(str(params.get("engine", "")))
        if auto_follow:
            self._worker_mode = "auto_follow"
            self.status_label.setText(f"[{engine_text}] Auto-follow: refreshing current-window detection...")
        else:
            self._worker_mode = "manual"
            self._set_running(True)
            self.progress.setValue(0)
            if current_window_only:
                self.status_label.setText(f"[{engine_text}] Starting current-window extraction...")
            else:
                self.status_label.setText(f"[{engine_text}] Starting full-hour auto extraction...")

        self.worker_thread = QThread(self)
        self.worker = ExtractWorker(self.backend, params)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.failed.connect(self.on_worker_failed)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)

        self.worker_thread.start()

    def _cleanup_worker(self) -> None:
        mode = self._worker_mode
        self._worker_mode = "idle"
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None
        if mode == "manual":
            self._set_running(False)
        else:
            self._update_window_slider()
        if self._auto_extract_pending and self.auto_scroll_check.isChecked() and self.worker_thread is None:
            self._auto_extract_pending = False
            self.start_extract(current_window_only=True, auto_follow=True)

    def on_worker_progress(self, pct: int, msg: str) -> None:
        if self._worker_mode == "auto_follow":
            return
        self.progress.setValue(max(0, min(100, int(pct))))
        self.status_label.setText(str(msg))

    def on_worker_finished(self, summary: dict) -> None:
        track_count = int(summary.get("track_count", 0))
        total_points = int(summary.get("total_points", 0))
        elapsed = float(summary.get("elapsed_seconds", float("nan")))
        params = summary.get("params", {})
        scope = str(params.get("scope", "full_data"))
        scope_text = "Current Window" if scope == "current_window" else "Full Hour"
        engine_text = str(params.get("engine_text", self._engine_text(str(params.get("engine", "")))))
        ext_cfg = params.get("extractor_config", {})
        if str(params.get("engine", "")) == "deep_learning":
            tmpl_text = "PeakSlotNet"
        else:
            tmpl_on = bool(ext_cfg.get("use_template_enhancement", True))
            tmpl_text = "Template ON" if tmpl_on else "Template OFF"
        if self._worker_mode == "auto_follow":
            window_end = (self.backend.current_start + self.backend.window_size) / self.backend.fs
            window_start = self.backend.current_start / self.backend.fs
            self.status_label.setText(
                f"[{engine_text} | {tmpl_text}] Auto-follow updated: "
                f"{track_count} tracks, {total_points} points, elapsed {elapsed:.2f}s, "
                f"window [{window_start:.1f}, {window_end:.1f}] s"
            )
            self.progress.setValue(0)
        else:
            self.progress.setValue(100)
            self.status_label.setText(
                f"[{engine_text} | {tmpl_text}] {scope_text} extraction completed: "
                f"{track_count} tracks, {total_points} points, elapsed {elapsed:.2f}s"
            )
        self.redraw()

    def on_worker_failed(self, message: str) -> None:
        self.status_label.setText(f"Extraction failed: {message}")
        if self._worker_mode != "auto_follow":
            QMessageBox.critical(self, "Auto Extraction Failed", message)

    def clear_results(self) -> None:
        self.backend.clear_tracks()
        self.progress.setValue(0)
        self.status_label.setText("Results cleared")
        self.redraw()

    def export_csv(self) -> None:
        default_path = str(Path(self.backend.files) / "auto_tracks.csv")
        out_csv, _ = QFileDialog.getSaveFileName(
            self,
            "Export Track CSV",
            default_path,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not out_csv:
            return
        try:
            csv_path, summary_path = self.backend.export_csv(out_csv)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        self.status_label.setText(f"Exported: {csv_path}\nSummary: {summary_path}")

    def _plot_section(self) -> None:
        data = self.backend.data_view
        if data.size == 0:
            return

        n_ch, n_samples = data.shape
        if n_ch <= 0 or n_samples <= 0:
            return

        decim = int(max(1, np.ceil(n_samples / float(self._plot_target_points_per_trace))))
        t = self.backend.t_axis_view[::decim]
        data_plot = np.asarray(data[:, ::decim], dtype=np.float64)
        offsets_km = np.asarray(self.backend.x_axis_m, dtype=np.float64) * 1e-3

        if offsets_km.size >= 2:
            spacing = float(np.median(np.diff(offsets_km)))
            if not np.isfinite(spacing) or spacing <= 0:
                spacing = 0.1
        else:
            spacing = 0.1
        wiggle_amp = 0.27 * spacing
        eps = 1e-12
        # Use one global gain for the whole window (no per-trace normalization).
        # Apply robust scaling with clipping to avoid abnormal spike amplification.
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
            # Keep all-zero channels visible as stable baselines in the current window.
            if (not np.isfinite(global_ref)) or global_ref < eps:
                x = np.full_like(t, offsets_km[i], dtype=np.float64)
            else:
                ratio = np.clip(trace / global_ref, -clip_ratio, clip_ratio)
                x = offsets_km[i] + ratio * wiggle_amp
            self.ax.plot(x, t, color="0.45", linewidth=0.8, alpha=0.9)

    def _plot_peak_slot_diagnostics(self) -> dict[str, int]:
        if self.show_peak_slot_diagnostics_check is None or (not self.show_peak_slot_diagnostics_check.isChecked()):
            return {"raw": 0, "final": 0, "removed": 0, "peaks": 0}
        diag = dict(getattr(self.backend, "last_peak_slot_diagnostics", {}) or {})
        if str(diag.get("model_family", "")) != "peak_slot":
            return {"raw": 0, "final": 0, "removed": 0, "peaks": 0}

        peak_index = np.asarray(diag.get("peak_index", []))
        peak_valid = np.asarray(diag.get("peak_valid", []))
        time_downsample = int(diag.get("time_downsample", 1) or 1)
        candidates = list(diag.get("candidates", []))
        peak_count = 0

        if peak_index.ndim == 2 and peak_valid.shape == peak_index.shape:
            peak_xs: list[float] = []
            peak_ts: list[float] = []
            for ch in range(int(peak_index.shape[0])):
                for k in range(int(peak_index.shape[1])):
                    if not bool(peak_valid[ch, k]):
                        continue
                    t_idx = int(peak_index[ch, k]) * int(max(1, time_downsample))
                    if not (0 <= t_idx < int(self.backend.window_size)):
                        continue
                    peak_xs.append(float(self.backend.x_axis_m[ch]) * 1e-3)
                    peak_ts.append(float(t_idx) / float(self.backend.fs))
                    peak_count += 1
            if peak_xs:
                self.ax.scatter(
                    np.asarray(peak_xs, dtype=np.float64),
                    np.asarray(peak_ts, dtype=np.float64),
                    s=8,
                    marker=".",
                    color="0.25",
                    alpha=0.18,
                    zorder=2,
                )

        raw_count = 0
        removed_count = 0
        final_count = 0
        for cand in candidates:
            points = list(cand.get("points", []))
            if len(points) < 2:
                continue
            xs = np.asarray([float(p["offset_km"]) for p in points], dtype=np.float64)
            ts = np.asarray([float(p["time_s"]) for p in points], dtype=np.float64)
            status = str(cand.get("status", "raw"))
            raw_count += 1
            if status == "final":
                final_count += 1
                self.ax.plot(xs, ts, color="#00a651", linewidth=1.0, alpha=0.85, linestyle="--", zorder=4)
                self.ax.scatter(xs, ts, s=24, facecolors="none", edgecolors="#00a651", linewidths=0.9, zorder=5)
            elif status != "removed_short":
                removed_count += 1
                self.ax.plot(xs, ts, color="#d62728", linewidth=0.8, alpha=0.28, linestyle="--", zorder=3)
                self.ax.scatter(xs, ts, s=12, marker="x", color="#d62728", alpha=0.28, zorder=3)

        return {
            "raw": int(raw_count),
            "final": int(final_count),
            "removed": int(removed_count),
            "peaks": int(peak_count),
        }

    def redraw(self) -> None:
        self.fig.clear()
        if self.backend.data_view.size == 0:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_axis_off()
            self.ax.text(
                0.5,
                0.5,
                "No data loaded\nPlease import a SAC data folder first",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            self._update_window_slider()
            self._sync_window_seconds_input()
            self.canvas.draw_idle()
            return

        self.ax = self.fig.add_subplot(111)
        self._plot_section()

        self.ax.set_ylim(self.backend.t_axis_view[0], self.backend.t_axis_view[-1])
        self.ax.invert_yaxis()
        y0 = self.backend.x_axis_m[0] * 1e-3
        y1 = self.backend.x_axis_m[-1] * 1e-3
        if self.backend.x_axis_m.size >= 2:
            dy = float(np.median(np.diff(self.backend.x_axis_m)) * 1e-3)
            if not np.isfinite(dy) or dy <= 0:
                dy = 0.1
        else:
            dy = 0.1
        pad = 0.4 * dy
        x_min_plot = y0 - pad
        x_max_plot = y1 + pad
        x_span_plot = max(1e-6, x_max_plot - x_min_plot)
        self.ax.set_xlim(x_min_plot, x_max_plot)
        self.ax.margins(x=0, y=0)
        self.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y + self.backend.t_offset:.1f}")
        )
        self.ax.set_xlabel("Offset [km]")
        self.ax.set_ylabel("Time [s]")

        track_count = len(self.backend.tracks)
        point_count = int(sum(len(t.points) for t in self.backend.tracks))
        cmap = cm.get_cmap("tab20", max(1, track_count))

        t0 = self.backend.current_start
        t1 = self.backend.current_start + self.backend.window_size
        for tr in self.backend.tracks:
            pts = [p for p in tr.points if t0 <= p.t_idx < t1]
            if len(pts) < 2:
                continue
            pts.sort(key=lambda p: p.ch_idx)
            ts = np.array([(p.t_idx - t0) / self.backend.fs for p in pts], dtype=np.float64)
            xs = np.array([p.offset_m * 1e-3 for p in pts], dtype=np.float64)
            color = cmap(tr.track_id % max(1, cmap.N))
            self.ax.plot(xs, ts, color=color, linewidth=1.1, alpha=0.95)
            self.ax.scatter(xs, ts, color=[color], s=10, marker="o", alpha=0.9)

            speed_kmh = float(tr.mean_speed_kmh)
            if not np.isfinite(speed_kmh):
                dt = np.diff(ts)
                dx_m = np.abs(np.diff(xs)) * 1000.0
                valid = np.abs(dt) > 1e-9
                if np.any(valid):
                    speed_kmh = float(np.mean(3.6 * dx_m[valid] / np.abs(dt[valid])))
            if np.isfinite(speed_kmh):
                mid = len(ts) // 2
                x_text = min(x_max_plot - 0.02 * x_span_plot, xs[mid] + 0.01 * x_span_plot)
                self.ax.text(
                    x_text,
                    ts[mid],
                    f"{speed_kmh:.1f} km/h",
                    color=color,
                    fontsize=8,
                    ha="left",
                    va="center",
                    alpha=0.95,
                    bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 0.8},
                )

        diag_counts = self._plot_peak_slot_diagnostics()
        window_end = (self.backend.current_start + self.backend.window_size) / self.backend.fs
        window_start = self.backend.current_start / self.backend.fs
        title = f"Tracks={track_count}, Points={point_count}, Window=[{window_start:.1f}, {window_end:.1f}] s"
        if self.show_peak_slot_diagnostics_check is not None and self.show_peak_slot_diagnostics_check.isChecked():
            title += (
                f" | PeakSlot diag: peaks={diag_counts['peaks']}, raw={diag_counts['raw']}, "
                f"final={diag_counts['final']}, removed={diag_counts['removed']}"
            )
        self.ax.set_title(title)
        self._sync_window_seconds_input()
        self._update_window_slider()
        self.canvas.draw_idle()

    def on_scroll(self, event) -> None:
        if event is None:
            return
        if getattr(self.toolbar, "mode", ""):
            return
        if hasattr(event, "inaxes") and event.inaxes is not self.ax:
            return
        step = getattr(event, "step", 0)
        if step == 0:
            step = 1 if getattr(event, "button", None) == "up" else -1
        step = 1 if step > 0 else -1

        modifiers = None
        if hasattr(event, "guiEvent") and event.guiEvent is not None:
            try:
                modifiers = event.guiEvent.modifiers()
            except Exception:  # noqa: BLE001
                modifiers = None
        zoom = bool(modifiers and (modifiers & Qt.KeyboardModifier.ControlModifier))
        if zoom and self.lock_window_check.isChecked():
            return
        if not zoom and self._switch_tensor_shard_sample(delta=-step):
            return
        cursor_t = float(event.ydata) if event.ydata is not None else None
        if cursor_t is not None and self.backend.t_axis_view.size > 0:
            cursor_t = max(0.0, min(cursor_t, float(self.backend.t_axis_view[-1])))

        changed = self.backend.handle_scroll(step, zoom, cursor_t)
        if changed:
            self.redraw()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vehicle Trajectory Auto-Extraction GUI (Peak Map + Dynamic Programming)")
    parser.add_argument(
        "--data-folder",
        type=str,
        default=DEFAULT_DATA_FOLDER,
        help="SAC data folder (default: synthetic_sac)",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    gui = AutoTrackGUI(data_folder=args.data_folder)
    gui.resize(1700, 900)
    gui.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
