from __future__ import annotations

import csv
import json
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional
import xml.etree.ElementTree as ET

import numpy as np
from obspy import Stream, Trace, read

from autotrack.core.track_extractor_graph import ExtractorConfig, Track, TrackPoint, extract_all


DEFAULT_DATA_FOLDER = "/Volumes/SanDisk2T4/MyProjects/BaFang/KF/data/synthetic_sac"


class AutoTrackBackend:
    def __init__(
        self,
        data_folder: str = DEFAULT_DATA_FOLDER,
        window_size: int = 1000 * 240,
        scroll_step_s: float = 30.0,
    ):
        self.files = str(Path(data_folder))
        self.window_size = int(window_size)
        self.default_window_size = int(window_size)
        self.scroll_step_s = float(scroll_step_s)
        self.current_start = 0

        self.data_all = np.empty((0, 0), dtype=np.float32)
        self.data_view = np.empty((0, 0), dtype=np.float32)
        self.x_axis_m = np.empty((0,), dtype=np.float64)
        self.t_axis_view = np.empty((0,), dtype=np.float64)
        self.fs = 1000.0
        self.dt = 0.001
        self.dx_m = 100.0
        self.t_offset = 0.0
        self.scroll_step_samples = 1

        self.tracks: list[Track] = []
        self.last_summary: dict = {}
        self.last_import_info: dict = {"position_sort_enabled": False}

        self.st_visual = Stream()
        self.init_error: Optional[str] = None

        try:
            self._load_data_all()
            self.update_view_window()
        except Exception as exc:  # noqa: BLE001
            self.init_error = str(exc)

    @staticmethod
    def _trace_distance(tr: Trace, fallback: float) -> float:
        dist = getattr(tr.stats, "distance", fallback)
        try:
            return float(dist)
        except Exception:  # noqa: BLE001
            return float(fallback)

    @staticmethod
    def _extract_device_id_from_filename(filename: str) -> Optional[str]:
        name = Path(filename).name.upper()
        match = re.search(r"_([0-9A-Z]{8})_EH", name)
        if match:
            return match.group(1)
        match = re.search(r"_([0-9A-Z]{8})_", name)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _parse_position_value(raw: object) -> Optional[float]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            try:
                return float(raw)
            except Exception:  # noqa: BLE001
                return None
        text = str(raw).strip()
        if not text:
            return None
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if not match:
            return None
        value = float(match.group(1))
        if not match.group(1).startswith("-"):
            if "南" in text:
                value = -abs(value)
            elif "北" in text:
                value = abs(value)
        return value

    @staticmethod
    def _xlsx_col_to_index(ref: str) -> Optional[int]:
        match = re.match(r"([A-Z]+)", ref)
        if not match:
            return None
        idx = 0
        for ch in match.group(1):
            idx = idx * 26 + (ord(ch) - 64)
        return idx

    def _load_position_map_from_xlsx(self, xlsx_path: str) -> dict[str, float]:
        xlsx = Path(xlsx_path).expanduser()
        if not xlsx.exists():
            raise FileNotFoundError(f"Position XLSX not found: {xlsx}")
        if not xlsx.is_file():
            raise FileNotFoundError(f"Position XLSX is not a file: {xlsx}")

        ns = {
            "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        rel_ns = {"p": "http://schemas.openxmlformats.org/package/2006/relationships"}

        with zipfile.ZipFile(xlsx, "r") as zf:
            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in zf.namelist():
                sst = ET.fromstring(zf.read("xl/sharedStrings.xml"))
                for si in sst.findall("m:si", ns):
                    pieces = [t.text or "" for t in si.findall(".//m:t", ns)]
                    shared_strings.append("".join(pieces))

            workbook = ET.fromstring(zf.read("xl/workbook.xml"))
            wb_rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
            rel_map = {
                rel.attrib.get("Id"): rel.attrib.get("Target", "")
                for rel in wb_rels.findall("p:Relationship", rel_ns)
            }

            first_sheet = workbook.find("m:sheets/m:sheet", ns)
            if first_sheet is None:
                raise ValueError(f"No worksheet found in position XLSX: {xlsx}")
            rid = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = rel_map.get(rid, "")
            if not target:
                raise ValueError(f"Cannot locate worksheet target in position XLSX: {xlsx}")
            sheet_xml_path = target if target.startswith("xl/") else f"xl/{target}"
            sheet = ET.fromstring(zf.read(sheet_xml_path))

            def _cell_value(cell: ET.Element) -> Optional[str]:
                typ = cell.attrib.get("t")
                val = cell.find("m:v", ns)
                if val is None:
                    inline = cell.find("m:is/m:t", ns)
                    return inline.text if inline is not None else None
                raw = val.text
                if typ == "s" and raw is not None:
                    idx = int(raw)
                    return shared_strings[idx] if 0 <= idx < len(shared_strings) else raw
                return raw

            rows: list[tuple[int, dict[int, Optional[str]]]] = []
            for row in sheet.findall("m:sheetData/m:row", ns):
                row_idx = int(row.attrib.get("r", "0"))
                row_data: dict[int, Optional[str]] = {}
                for cell in row.findall("m:c", ns):
                    ref = cell.attrib.get("r", "")
                    col_idx = self._xlsx_col_to_index(ref)
                    if col_idx is None:
                        continue
                    row_data[col_idx] = _cell_value(cell)
                if row_data:
                    rows.append((row_idx, row_data))

        header_row_idx: Optional[int] = None
        position_cols: list[int] = []
        for row_idx, row_data in rows:
            cols = [
                col
                for col, value in row_data.items()
                if isinstance(value, str) and ("位置" in value)
            ]
            if cols:
                header_row_idx = row_idx
                position_cols = sorted(cols)
                break
        if header_row_idx is None or not position_cols:
            raise ValueError(f"Cannot find '位置' column in position XLSX: {xlsx}")

        id_to_position_km: dict[str, float] = {}
        for row_idx, row_data in rows:
            if row_idx <= header_row_idx:
                continue
            loc_raw: Optional[object] = None
            for col in position_cols:
                value = row_data.get(col)
                if value is None:
                    continue
                if str(value).strip():
                    loc_raw = value
                    break
            pos_km = self._parse_position_value(loc_raw)
            if pos_km is None:
                continue

            row_ids: set[str] = set()
            for value in row_data.values():
                if value is None:
                    continue
                text = str(value).strip().upper()
                if not text:
                    continue
                for token in re.findall(r"[0-9A-Z]{8}", text):
                    if token.startswith("19"):
                        row_ids.add(token)
            for device_id in row_ids:
                id_to_position_km.setdefault(device_id, float(pos_km))

        if not id_to_position_km:
            raise ValueError(f"No device-position mapping found in position XLSX: {xlsx}")
        return id_to_position_km

    def _reorder_traces_by_position_xlsx(
        self,
        traces_with_files: list[tuple[Trace, Path]],
        xlsx_path: str,
    ) -> tuple[list[Trace], np.ndarray, dict]:
        id_to_position_km = self._load_position_map_from_xlsx(xlsx_path)

        matched: list[tuple[float, int, Trace]] = []
        unmatched: list[tuple[float, int, Trace]] = []
        unmatched_ids: list[str] = []

        for idx, (trace, file_path) in enumerate(traces_with_files):
            device_id = self._extract_device_id_from_filename(file_path.name)
            if device_id and (device_id in id_to_position_km):
                matched.append((float(id_to_position_km[device_id]) * 1000.0, idx, trace))
                continue
            unmatched.append((self._trace_distance(trace, idx), idx, trace))
            unmatched_ids.append(device_id or file_path.name)

        if not matched:
            raise ValueError(
                "Position XLSX enabled, but no SAC device IDs matched this table. "
                "Please check the selected XLSX file."
            )

        matched.sort(key=lambda x: (x[0], x[1]))
        unmatched.sort(key=lambda x: (x[0], x[1]))

        ordered_traces: list[Trace] = []
        x_axis: list[float] = []
        last_x = -np.inf
        for offset_m, _, trace in matched:
            x = float(offset_m)
            if not np.isfinite(x):
                x = (float(last_x) + 100.0) if np.isfinite(last_x) else 0.0
            if np.isfinite(last_x) and x <= last_x:
                x = float(last_x) + 1e-3
            ordered_traces.append(trace)
            x_axis.append(x)
            last_x = x
        for guessed_x, _, trace in unmatched:
            x = float(guessed_x)
            if not np.isfinite(x):
                x = (float(last_x) + 100.0) if np.isfinite(last_x) else 0.0
            if np.isfinite(last_x) and x <= last_x:
                x = float(last_x) + 100.0
            ordered_traces.append(trace)
            x_axis.append(x)
            last_x = x

        info = {
            "position_sort_enabled": True,
            "position_xlsx_path": str(Path(xlsx_path).expanduser()),
            "total_channels": int(len(traces_with_files)),
            "matched_channels": int(len(matched)),
            "unmatched_channels": int(len(unmatched)),
            "unmatched_examples": unmatched_ids[:8],
        }
        return ordered_traces, np.asarray(x_axis, dtype=np.float64), info

    def _read_data_all(
        self,
        folder: str,
        use_position_xlsx: bool = False,
        position_xlsx_path: Optional[str] = None,
    ) -> tuple[np.ndarray, float, np.ndarray, float, dict]:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Data folder does not exist: {folder_path}")
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        sac_files = sorted(folder_path.glob("*.sac"))
        if not sac_files:
            raise FileNotFoundError(f"No .sac files found in folder: {folder_path}")

        traces_with_files: list[tuple[Trace, Path]] = []
        for sac_file in sac_files:
            st = read(str(sac_file))
            for tr in st:
                traces_with_files.append((tr, sac_file))
        if not traces_with_files:
            raise FileNotFoundError(f"Read failed: no valid SAC files in folder: {folder_path}")

        import_info = {"position_sort_enabled": False}
        if use_position_xlsx:
            if not position_xlsx_path:
                raise ValueError("Position XLSX path is required when position sorting is enabled.")
            traces, x_axis, import_info = self._reorder_traces_by_position_xlsx(
                traces_with_files=traces_with_files,
                xlsx_path=position_xlsx_path,
            )
        else:
            traces = [
                t
                for _, _, t in sorted(
                    (self._trace_distance(t, i), i, t)
                    for i, (t, _) in enumerate(traces_with_files)
                )
            ]
            x_axis = np.array([self._trace_distance(tr, i) for i, tr in enumerate(traces)], dtype=np.float64)

        data = np.vstack([np.asarray(tr.data, dtype=np.float32) for tr in traces])

        deltas = np.array([float(getattr(tr.stats, "delta", 0.001)) for tr in traces], dtype=np.float64)
        delta = float(np.median(deltas))
        if delta <= 0:
            raise ValueError("Invalid SAC sampling interval")
        fs = 1.0 / delta

        if x_axis.size >= 2 and np.any(np.diff(x_axis) <= 0):
            # Fallback to equal spacing when distance headers are unreliable
            dx = 100.0
            x_axis = np.arange(len(traces), dtype=np.float64) * dx
        dx_m = float(np.median(np.diff(x_axis))) if x_axis.size >= 2 else 100.0
        if dx_m <= 0:
            dx_m = 100.0
        # In many SAC files, distance may be missing (0,1,2... by channel index) or stored in km (0.1 means 100 m).
        # This project defaults to 100 m channel spacing, so abnormally small spacing is uniformly corrected.
        if dx_m < 10.0:
            dx_m = 100.0
            x_axis = np.arange(len(traces), dtype=np.float64) * dx_m

        return data, fs, x_axis, dx_m, import_info

    def _load_data_all(
        self,
        use_position_xlsx: bool = False,
        position_xlsx_path: Optional[str] = None,
    ) -> None:
        data, fs, x_axis, dx_m, import_info = self._read_data_all(
            self.files,
            use_position_xlsx=use_position_xlsx,
            position_xlsx_path=position_xlsx_path,
        )
        self.data_all = data
        self.fs = float(fs)
        self.dt = 1.0 / self.fs
        self.x_axis_m = x_axis
        self.dx_m = float(dx_m)
        self.last_import_info = dict(import_info)

    @staticmethod
    def _build_stream(
        data: np.ndarray,
        x_axis_m: np.ndarray,
        t_axis: np.ndarray,
        starttime_offset: float = 0.0,
    ) -> Stream:
        st = Stream()
        if t_axis.size < 2:
            return st
        dt = float(t_axis[1] - t_axis[0])
        for i, d in enumerate(data):
            tr = Trace(data=np.asarray(d, dtype=np.float32))
            tr.stats.distance = float(x_axis_m[i])
            tr.stats.starttime = float(t_axis[0] + starttime_offset)
            tr.stats.delta = dt
            st.append(tr)
        return st

    def load_data_folder(
        self,
        data_folder: str,
        reset_results: bool = True,
        use_position_xlsx: bool = False,
        position_xlsx_path: Optional[str] = None,
    ) -> bool:
        folder = Path(data_folder).expanduser()
        self.files = str(folder)
        self.current_start = 0
        self.window_size = self.default_window_size
        self._load_data_all(
            use_position_xlsx=use_position_xlsx,
            position_xlsx_path=position_xlsx_path,
        )
        if reset_results:
            self.clear_tracks()
        self.update_view_window()
        self.init_error = None
        return True

    def update_view_window(self) -> bool:
        if self.data_all.size == 0:
            return False
        max_samples = self.data_all.shape[1]
        self.window_size = max(1000, min(self.window_size, max_samples))
        max_start = max(0, max_samples - self.window_size)
        self.current_start = max(0, min(self.current_start, max_start))

        end = self.current_start + self.window_size
        self.data_view = self.data_all[:, self.current_start:end]
        n_view = self.data_view.shape[1]
        self.t_axis_view = np.arange(n_view, dtype=np.float64) / self.fs
        self.t_offset = self.current_start / self.fs
        # Adapt scroll step to window length to avoid oversized jumps on short windows.
        base_step = max(1, int(round(self.scroll_step_s * self.fs)))
        adaptive_step = max(1, int(round(0.1 * self.window_size)))
        self.scroll_step_samples = min(base_step, adaptive_step)
        return True

    def adjust_window_size(self, scale: float) -> bool:
        if self.data_all.size == 0:
            return False
        new_size = int(self.window_size * scale)
        new_size = max(1000, min(new_size, self.data_all.shape[1]))
        if new_size == self.window_size:
            return False
        self.window_size = new_size
        self.current_start = min(self.current_start, max(0, self.data_all.shape[1] - self.window_size))
        return self.update_view_window()

    def adjust_window_size_at_cursor(self, scale: float, cursor_x: Optional[float]) -> bool:
        if cursor_x is None:
            return self.adjust_window_size(scale)
        if self.data_all.size == 0:
            return False

        old_size = self.window_size
        new_size = int(self.window_size * scale)
        new_size = max(1000, min(new_size, self.data_all.shape[1]))
        if new_size == old_size:
            return False

        cursor_global_s = float(cursor_x) + self.t_offset
        old_window_s = old_size / self.fs
        rel = 0.0 if old_window_s <= 0 else float(cursor_x) / old_window_s
        new_start_s = cursor_global_s - rel * (new_size / self.fs)
        new_start = int(round(new_start_s * self.fs))
        new_start = max(0, min(new_start, self.data_all.shape[1] - new_size))

        self.window_size = new_size
        self.current_start = new_start
        return self.update_view_window()

    def handle_scroll(self, step: int, zoom: bool, cursor_x: Optional[float]) -> bool:
        if self.data_all.size == 0 or step == 0:
            return False
        step = 1 if step > 0 else -1
        if zoom:
            scale = 0.8 if step > 0 else 1.25
            return self.adjust_window_size_at_cursor(scale, cursor_x)

        shift = int(self.scroll_step_samples) * (-step)
        new_start = self.current_start + shift
        new_start = max(0, min(new_start, self.data_all.shape[1] - self.window_size))
        if new_start == self.current_start:
            return False
        self.current_start = new_start
        return self.update_view_window()

    @staticmethod
    def _compute_point_speed_kmh(p1: TrackPoint, p2: TrackPoint) -> float:
        dt = abs(float(p2.time_s) - float(p1.time_s))
        dx = abs(float(p2.offset_m) - float(p1.offset_m))
        if dt <= 1e-9 or dx <= 1e-9:
            return float("nan")
        return float(3.6 * dx / dt)

    @classmethod
    def _local_speed_series(cls, points: list[TrackPoint]) -> list[float]:
        n = len(points)
        if n == 0:
            return []
        speeds = [float("nan")] * n
        for i in range(n):
            vals = []
            if i - 1 >= 0:
                vals.append(cls._compute_point_speed_kmh(points[i - 1], points[i]))
            if i + 1 < n:
                vals.append(cls._compute_point_speed_kmh(points[i], points[i + 1]))
            vals = [v for v in vals if np.isfinite(v)]
            speeds[i] = float(np.mean(vals)) if vals else float("nan")
        return speeds

    @staticmethod
    def _track_points_map(track: Track) -> dict[int, TrackPoint]:
        return {p.ch_idx: p for p in track.points}

    def _recompute_track_stats(self, track_id: int, direction: str, points: list[TrackPoint]) -> Track:
        points_sorted = sorted(points, key=lambda p: p.ch_idx)
        total_score = float(np.sum([p.score for p in points_sorted]))
        speeds = self._local_speed_series(points_sorted)
        valid = [v for v in speeds if np.isfinite(v)]
        mean_speed = float(np.mean(valid)) if valid else float("nan")
        return Track(
            track_id=track_id,
            direction=direction,
            points=points_sorted,
            total_score=total_score,
            mean_speed_kmh=mean_speed,
        )

    @staticmethod
    def _track_overlap(existing: Track, new: Track, tol_samples: int) -> tuple[int, float]:
        a = {p.ch_idx: p.t_idx for p in existing.points}
        b = {p.ch_idx: p.t_idx for p in new.points}
        common = sorted(set(a.keys()) & set(b.keys()))
        if not common:
            return 0, float("inf")
        diffs = np.array([abs(int(a[ch]) - int(b[ch])) for ch in common], dtype=np.float64)
        matched = diffs <= float(tol_samples)
        if not np.any(matched):
            return 0, float(np.median(diffs))
        return int(np.sum(matched)), float(np.median(diffs[matched]))

    def _merge_two_tracks(self, existing: Track, new: Track, tol_samples: int) -> Track:
        points = self._track_points_map(existing)
        for p in new.points:
            old = points.get(p.ch_idx)
            if old is None:
                points[p.ch_idx] = p
                continue
            if abs(int(old.t_idx) - int(p.t_idx)) <= tol_samples:
                points[p.ch_idx] = p if p.score >= old.score else old
            else:
                points[p.ch_idx] = p if p.score > old.score else old
        return self._recompute_track_stats(existing.track_id, existing.direction, list(points.values()))

    def _merge_tracks(self, tracks: list[Track], tol_samples: int, min_overlap: int = 3) -> list[Track]:
        if not tracks:
            return []
        ordered = sorted(
            tracks,
            key=lambda tr: min((p.t_idx for p in tr.points), default=10**12),
        )
        merged: list[Track] = []
        for tr in ordered:
            best_idx = -1
            best_overlap = 0
            best_med = float("inf")
            for i, ex in enumerate(merged):
                overlap, med = self._track_overlap(ex, tr, tol_samples)
                if overlap > best_overlap or (overlap == best_overlap and med < best_med):
                    best_overlap = overlap
                    best_med = med
                    best_idx = i
            if best_idx >= 0 and best_overlap >= min_overlap:
                merged[best_idx] = self._merge_two_tracks(merged[best_idx], tr, tol_samples)
            else:
                merged.append(self._recompute_track_stats(tr.track_id, tr.direction, tr.points))
        return merged

    def _deduplicate_tracks(self, tracks: list[Track], tol_samples: int) -> list[Track]:
        if not tracks:
            return []
        ordered = sorted(tracks, key=lambda tr: tr.total_score, reverse=True)
        kept: list[Track] = []
        for tr in ordered:
            duplicate = False
            for ex in kept:
                overlap, _ = self._track_overlap(ex, tr, tol_samples)
                if overlap <= 0:
                    continue
                ratio = overlap / max(1, min(len(ex.points), len(tr.points)))
                speed_diff = abs(float(ex.mean_speed_kmh) - float(tr.mean_speed_kmh))
                speed_ok = (not np.isfinite(speed_diff)) or speed_diff <= 15.0
                if ratio >= 0.6 and speed_ok:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(tr)
        return kept

    @staticmethod
    def _track_channel_bounds(track: Track) -> tuple[Optional[TrackPoint], Optional[TrackPoint], set[int]]:
        if not track.points:
            return None, None, set()
        pts = sorted(track.points, key=lambda p: p.ch_idx)
        return pts[0], pts[-1], {int(p.ch_idx) for p in pts}

    @staticmethod
    def _dt_link_bounds(
        direction: str,
        delta_x_m: float,
        vmin_mps: float,
        vmax_mps: float,
        slack_ratio: float,
    ) -> tuple[float, float]:
        slack = max(0.0, float(slack_ratio))
        if direction == "forward":
            lo = (delta_x_m / vmax_mps) * max(0.0, 1.0 - slack)
            hi = (delta_x_m / vmin_mps) * (1.0 + slack)
        else:
            base_lo = -(delta_x_m / vmin_mps)
            base_hi = -(delta_x_m / vmax_mps)
            lo = base_lo * (1.0 + slack)
            hi = base_hi * max(0.0, 1.0 - slack)
        return (lo, hi) if lo <= hi else (hi, lo)

    def _stitch_link_cost(
        self,
        left: Track,
        right: Track,
        direction: str,
        speed_min_kmh: float,
        speed_max_kmh: float,
        max_gap_channels: int,
        dt_slack_ratio: float,
        max_speed_diff_kmh: float,
    ) -> Optional[float]:
        if left.direction != direction or right.direction != direction:
            return None
        l_start, l_end, l_set = self._track_channel_bounds(left)
        r_start, _, r_set = self._track_channel_bounds(right)
        if l_end is None or r_start is None:
            return None
        if l_set & r_set:
            return None

        dch = int(r_start.ch_idx) - int(l_end.ch_idx)
        if dch < 1 or dch > int(max_gap_channels):
            return None

        dx = abs(float(r_start.offset_m) - float(l_end.offset_m))
        if dx <= 1e-9:
            dx = float(dch) * float(self.dx_m)
        if dx <= 1e-9:
            return None

        dt = float(r_start.time_s) - float(l_end.time_s)
        vmin_mps = float(speed_min_kmh) / 3.6
        vmax_mps = float(speed_max_kmh) / 3.6
        dt_lo, dt_hi = self._dt_link_bounds(direction, dx, vmin_mps, vmax_mps, dt_slack_ratio)
        if not (dt_lo <= dt <= dt_hi):
            return None

        speed_link_kmh = 3.6 * dx / max(1e-9, abs(dt))
        low_speed_allow = max(1e-6, float(speed_min_kmh) * 0.65)
        high_speed_allow = float(speed_max_kmh) * 1.35
        if not (low_speed_allow <= speed_link_kmh <= high_speed_allow):
            return None

        speed_diff = 0.0
        if np.isfinite(left.mean_speed_kmh) and np.isfinite(right.mean_speed_kmh):
            speed_diff = abs(float(left.mean_speed_kmh) - float(right.mean_speed_kmh))
            if speed_diff > float(max_speed_diff_kmh):
                return None

        dt_mid = 0.5 * (dt_lo + dt_hi)
        dt_scale = max(1e-6, abs(dt_hi - dt_lo))
        dt_cost = abs(dt - dt_mid) / dt_scale
        speed_cost = speed_diff / max(1e-6, float(max_speed_diff_kmh))
        gap_cost = float(dch)
        return gap_cost + 0.6 * dt_cost + 0.8 * speed_cost

    def _stitch_track_fragments(
        self,
        tracks: list[Track],
        direction: str,
        speed_min_kmh: float,
        speed_max_kmh: float,
        tol_samples: int,
        max_gap_channels: int = 8,
        dt_slack_ratio: float = 0.35,
        max_speed_diff_kmh: float = 35.0,
    ) -> list[Track]:
        if len(tracks) < 2:
            return tracks

        merged = [self._recompute_track_stats(i, tr.direction, tr.points) for i, tr in enumerate(tracks)]
        while True:
            best_pair: Optional[tuple[int, int]] = None
            best_cost = float("inf")
            n = len(merged)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    cost = self._stitch_link_cost(
                        left=merged[i],
                        right=merged[j],
                        direction=direction,
                        speed_min_kmh=speed_min_kmh,
                        speed_max_kmh=speed_max_kmh,
                        max_gap_channels=max_gap_channels,
                        dt_slack_ratio=dt_slack_ratio,
                        max_speed_diff_kmh=max_speed_diff_kmh,
                    )
                    if cost is None:
                        continue
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (i, j)
            if best_pair is None:
                break

            i, j = best_pair
            stitched = self._merge_two_tracks(merged[i], merged[j], tol_samples=tol_samples)
            new_tracks: list[Track] = []
            for k, tr in enumerate(merged):
                if k in {i, j}:
                    continue
                new_tracks.append(tr)
            new_tracks.append(stitched)
            merged = new_tracks

        return [self._recompute_track_stats(i, tr.direction, tr.points) for i, tr in enumerate(merged)]

    def _to_global_track(self, track: Track, start_sample: int) -> Track:
        points = []
        for p in track.points:
            global_t = int(p.t_idx + start_sample)
            offset_m = float(self.x_axis_m[p.ch_idx])
            points.append(
                TrackPoint(
                    ch_idx=int(p.ch_idx),
                    t_idx=global_t,
                    time_s=global_t / self.fs,
                    offset_m=offset_m,
                    amp=float(p.amp),
                    score=float(p.score),
                )
            )
        return self._recompute_track_stats(track.track_id, track.direction, points)

    def _choose_enhance_decimate(self, n_samples: int) -> int:
        if n_samples >= int(round(300.0 * self.fs)):
            return 4
        if n_samples >= int(round(180.0 * self.fs)):
            return 3
        return 2

    def run_auto_extract(
        self,
        direction: str,
        speed_min_kmh: float,
        speed_max_kmh: float,
        prominence: float,
        min_peak_distance: int,
        min_track_channels: int,
        edge_min_track_channels: int = 4,
        edge_time_margin_seconds: float = 15.0,
        edge_min_score_scale: float = 0.2,
        tile_seconds: float = 120.0,
        overlap_seconds: float = 20.0,
        nms_time_radius: float = 0.18,
        current_window_only: bool = False,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        engine: str = "cpu_parallel",
        parallel_workers: Optional[int] = None,
        gpu_device_id: int = 0,
        enable_template_enhancement: bool = False,
        dl_model_path: str = "",
        dl_device: str = "",
        dl_objectness_threshold: float = 0.5,
        dl_visibility_threshold: float = 0.5,
        dl_min_visible_channels: int = 3,
        dl_refine_radius_samples: int = 120,
    ) -> dict:
        if self.data_all.size == 0:
            raise RuntimeError("No data loaded; cannot run auto extraction")
        if direction not in {"forward", "reverse"}:
            raise ValueError("direction must be either forward or reverse")
        if speed_min_kmh <= 0 or speed_max_kmh <= 0:
            raise ValueError("speed_min/speed_max must be > 0")
        if speed_min_kmh > speed_max_kmh:
            raise ValueError("speed_min(km/h) cannot be greater than speed_max(km/h)")
        if edge_min_track_channels < 2:
            raise ValueError("edge_min_track_channels must be >= 2")
        if edge_time_margin_seconds < 0:
            raise ValueError("edge_time_margin_seconds cannot be negative")
        if edge_min_score_scale < 0:
            raise ValueError("edge_min_score_scale cannot be negative")
        if engine not in {"cpu_single", "cpu_parallel", "gpu", "gpu_torch_mps", "deep_learning"}:
            raise ValueError("engine must be cpu_single / cpu_parallel / gpu / gpu_torch_mps / deep_learning")
        if engine == "deep_learning" and not str(dl_model_path).strip():
            raise ValueError("Deep Learning engine requires a model checkpoint path")
        if not current_window_only:
            if tile_seconds <= 0:
                raise ValueError("tile_seconds must be > 0")
            if overlap_seconds < 0:
                raise ValueError("overlap_seconds cannot be negative")

        t0 = time.perf_counter()
        nms_samples = int(max(1, round(float(nms_time_radius) * self.fs)))
        if current_window_only and self.data_view.size > 0:
            scope_samples = int(self.data_view.shape[1])
        else:
            scope_samples = int(max(round(tile_seconds * self.fs), 10 * self.fs))
        enhance_decimate = self._choose_enhance_decimate(scope_samples)
        cfg = ExtractorConfig(
            use_template_enhancement=bool(enable_template_enhancement),
            enhance_decimate=int(enhance_decimate),
            prominence=float(prominence),
            min_peak_distance=int(max(1, min_peak_distance)),
            min_track_channels=int(max(2, min_track_channels)),
            edge_min_track_channels=int(max(2, edge_min_track_channels)),
            edge_time_margin_seconds=float(max(0.0, edge_time_margin_seconds)),
            edge_min_score_scale=float(max(0.0, edge_min_score_scale)),
            nms_time_radius=nms_samples,
        )
        extract_config: ExtractorConfig | dict = cfg

        extract_fn = extract_all
        engine_text = "CPU Single-thread"
        workers = 1
        device_id_used: Optional[int] = None

        if engine == "cpu_parallel":
            workers = max(1, int(parallel_workers or (os.cpu_count() or 1)))
            engine_text = f"CPU Parallel({workers} workers)"
        elif engine == "gpu":
            try:
                import cupy as cp
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "No usable CuPy detected. Install a CuPy build matching your CUDA version, e.g. cupy-cuda12x."
                ) from exc
            n_dev = int(cp.cuda.runtime.getDeviceCount())
            if n_dev <= 0:
                raise RuntimeError("No CUDA devices detected")
            if int(gpu_device_id) < 0 or int(gpu_device_id) >= n_dev:
                raise ValueError(f"gpu_device_id out of range; available device count: {n_dev}")
            cp.cuda.Device(int(gpu_device_id)).use()
            from autotrack.cli.auto_track_gpu import extract_all_gpu

            extract_fn = extract_all_gpu
            device_id_used = int(gpu_device_id)
            engine_text = f"GPU(cuda:{device_id_used})"
        elif engine == "gpu_torch_mps":
            from autotrack.core.auto_track_torch_mps import extract_all_torch_mps

            extract_fn = extract_all_torch_mps
            engine_text = "GPU(PyTorch MPS)"
        elif engine == "deep_learning":
            from autotrack.core.trajectory_deep_engine import extract_all_deep_learning

            extract_fn = extract_all_deep_learning
            workers = 1
            engine_text = "Deep Learning(Trajectory Queries)"
            extract_config = {
                "model_path": str(dl_model_path).strip(),
                "device": str(dl_device).strip() or None,
                "objectness_threshold": float(dl_objectness_threshold),
                "visibility_threshold": float(dl_visibility_threshold),
                "min_visible_channels": int(max(1, dl_min_visible_channels)),
                "refine_radius_samples": int(max(0, dl_refine_radius_samples)),
                "max_tracks": int(cfg.max_tracks),
                "dedup_tolerance_samples": int(nms_samples),
            }

        if current_window_only:
            if self.data_view.size == 0:
                self.update_view_window()
            source = self.data_view
            start_sample = int(self.current_start)
            if progress_cb:
                progress_cb(0, f"[{engine_text}] Starting current-window extraction...")
            window_tracks = extract_fn(
                data=source,
                fs=self.fs,
                dx_m=self.dx_m,
                direction=direction,
                vmin_kmh=float(speed_min_kmh),
                vmax_kmh=float(speed_max_kmh),
                config=extract_config,
            )
            if progress_cb:
                progress_cb(75, "Deduplicating window tracks...")
            global_tracks = [self._to_global_track(tr, start_sample) for tr in window_tracks]
            dedup = self._deduplicate_tracks(global_tracks, tol_samples=nms_samples)
            if progress_cb:
                progress_cb(84, "Stitching window tracks...")
            dedup = self._stitch_track_fragments(
                dedup,
                direction=direction,
                speed_min_kmh=float(speed_min_kmh),
                speed_max_kmh=float(speed_max_kmh),
                tol_samples=nms_samples,
            )
            dedup = self._deduplicate_tracks(dedup, tol_samples=nms_samples)
            dedup = sorted(dedup, key=lambda tr: tr.total_score, reverse=True)
        else:
            tile_samples = int(max(round(tile_seconds * self.fs), 10 * self.fs))
            overlap_samples = int(max(0, round(overlap_seconds * self.fs)))
            overlap_samples = min(overlap_samples, tile_samples - 1)
            step = max(1, tile_samples - overlap_samples)

            n_total = self.data_all.shape[1]
            starts = list(range(0, n_total, step))
            all_tracks: list[Track] = []

            if progress_cb:
                progress_cb(0, f"[{engine_text}] Starting tiled extraction...")

            if engine == "cpu_parallel" and len(starts) > 1:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {}
                    for start in starts:
                        end = min(n_total, start + tile_samples)
                        tile = self.data_all[:, start:end]
                        fut = ex.submit(
                            extract_fn,
                            data=tile,
                            fs=self.fs,
                            dx_m=self.dx_m,
                            direction=direction,
                            vmin_kmh=float(speed_min_kmh),
                            vmax_kmh=float(speed_max_kmh),
                            config=extract_config,
                        )
                        futures[fut] = int(start)
                    done = 0
                    for fut in as_completed(futures):
                        start = futures[fut]
                        tile_tracks = fut.result()
                        all_tracks.extend(self._to_global_track(tr, start) for tr in tile_tracks)
                        done += 1
                        if progress_cb:
                            pct = int(round(70.0 * done / max(1, len(starts))))
                            progress_cb(pct, f"[{engine_text}] Tiled extraction {done}/{len(starts)}")
            else:
                for i, start in enumerate(starts):
                    end = min(n_total, start + tile_samples)
                    tile = self.data_all[:, start:end]
                    tile_tracks = extract_fn(
                        data=tile,
                        fs=self.fs,
                        dx_m=self.dx_m,
                        direction=direction,
                        vmin_kmh=float(speed_min_kmh),
                        vmax_kmh=float(speed_max_kmh),
                        config=extract_config,
                    )
                    all_tracks.extend(self._to_global_track(tr, start) for tr in tile_tracks)
                    if progress_cb:
                        pct = int(round(70.0 * (i + 1) / max(1, len(starts))))
                        progress_cb(pct, f"[{engine_text}] Tiled extraction {i + 1}/{len(starts)}")

            if progress_cb:
                progress_cb(78, f"[{engine_text}] Merging across tiles...")
            merged = self._merge_tracks(all_tracks, tol_samples=nms_samples, min_overlap=3)

            if progress_cb:
                progress_cb(88, f"[{engine_text}] Global deduplication...")
            dedup = self._deduplicate_tracks(merged, tol_samples=nms_samples)
            if progress_cb:
                progress_cb(93, f"[{engine_text}] Track stitching...")
            dedup = self._stitch_track_fragments(
                dedup,
                direction=direction,
                speed_min_kmh=float(speed_min_kmh),
                speed_max_kmh=float(speed_max_kmh),
                tol_samples=nms_samples,
            )
            dedup = self._deduplicate_tracks(dedup, tol_samples=nms_samples)
            dedup = sorted(dedup, key=lambda tr: tr.total_score, reverse=True)

        self.tracks = [
            Track(
                track_id=i,
                direction=tr.direction,
                points=tr.points,
                total_score=tr.total_score,
                mean_speed_kmh=tr.mean_speed_kmh,
            )
            for i, tr in enumerate(dedup)
        ]

        elapsed = float(time.perf_counter() - t0)
        total_points = int(sum(len(tr.points) for tr in self.tracks))
        avg_points = float(total_points / max(1, len(self.tracks)))
        if engine == "deep_learning":
            extractor_config_summary = {
                "model_path": str(dl_model_path).strip(),
                "device": str(dl_device).strip() or "auto",
                "objectness_threshold": float(dl_objectness_threshold),
                "visibility_threshold": float(dl_visibility_threshold),
                "min_visible_channels": int(dl_min_visible_channels),
                "refine_radius_samples": int(dl_refine_radius_samples),
                "dedup_tolerance_samples": int(nms_samples),
            }
        else:
            extractor_config_summary = {
                "sigma_seconds": list(cfg.sigma_seconds),
                "use_template_enhancement": bool(cfg.use_template_enhancement),
                "enhance_decimate": int(cfg.enhance_decimate),
                "max_skip_channels": int(cfg.max_skip_channels),
                "lambda_speed": float(cfg.lambda_speed),
                "lambda_skip": float(cfg.lambda_skip),
                "speed_change_tolerance_kmh": float(cfg.speed_change_tolerance_kmh),
                "speed_penalty_power": float(cfg.speed_penalty_power),
                "speed_penalty_cap": float(cfg.speed_penalty_cap),
                "min_track_score": float(cfg.min_track_score),
                "nms_channel_radius": int(cfg.nms_channel_radius),
            }
        self.last_summary = {
            "track_count": int(len(self.tracks)),
            "total_points": total_points,
            "avg_points_per_track": avg_points,
            "elapsed_seconds": elapsed,
            "params": {
                "direction": direction,
                "scope": "current_window" if current_window_only else "full_data",
                "engine": str(engine),
                "engine_text": engine_text,
                "parallel_workers": int(workers) if engine == "cpu_parallel" else None,
                "gpu_device_id": int(device_id_used) if device_id_used is not None else None,
                "speed_min_kmh": float(speed_min_kmh),
                "speed_max_kmh": float(speed_max_kmh),
                "prominence": float(prominence),
                "min_peak_distance": int(min_peak_distance),
                "min_track_channels": int(min_track_channels),
                "edge_min_track_channels": int(edge_min_track_channels),
                "edge_time_margin_seconds": float(edge_time_margin_seconds),
                "edge_min_score_scale": float(edge_min_score_scale),
                "tile_seconds": float(tile_seconds),
                "overlap_seconds": float(overlap_seconds),
                "nms_time_radius_seconds": float(nms_time_radius),
                "window_start_sample": int(self.current_start),
                "window_end_sample": int(self.current_start + self.window_size),
                "extractor_config": extractor_config_summary,
                "stitch_config": {
                    "max_gap_channels": 8,
                    "dt_slack_ratio": 0.35,
                    "max_speed_diff_kmh": 35.0,
                },
            },
        }
        if progress_cb:
            progress_cb(100, f"[{engine_text}] Extraction completed")
        return self.last_summary

    def clear_tracks(self) -> None:
        self.tracks = []
        self.last_summary = {}

    def export_csv(self, csv_path: Optional[str] = None) -> tuple[str, str]:
        if not self.tracks:
            raise RuntimeError("No tracks available to export")

        if csv_path:
            out_csv = Path(csv_path).expanduser()
        else:
            out_csv = Path(self.files) / "auto_tracks.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "track_id",
                    "direction",
                    "ch_idx",
                    "offset_m",
                    "t_idx",
                    "time_s",
                    "amp",
                    "score",
                    "local_speed_kmh",
                ]
            )
            for tr in self.tracks:
                pts = sorted(tr.points, key=lambda p: p.ch_idx)
                local_speed = self._local_speed_series(pts)
                for p, sp in zip(pts, local_speed):
                    writer.writerow(
                        [
                            int(tr.track_id),
                            tr.direction,
                            int(p.ch_idx),
                            float(p.offset_m),
                            int(p.t_idx),
                            float(p.time_s),
                            float(p.amp),
                            float(p.score),
                            float(sp) if np.isfinite(sp) else np.nan,
                        ]
                    )

        summary_path = out_csv.parent / "summary.json"
        summary = dict(self.last_summary) if self.last_summary else {}
        summary["csv_path"] = str(out_csv)
        summary["summary_path"] = str(summary_path)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return str(out_csv), str(summary_path)
