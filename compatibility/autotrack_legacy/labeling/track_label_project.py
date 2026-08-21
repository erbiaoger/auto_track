"""Editable track-label project storage for real DAS data.

Purpose:
    Provide one shared data layer for automatic track proposals and later manual
    calibration inside the real-data labeling GUI. The GUI should not manage raw
    JSON dicts directly. This module keeps an explicit editable representation,
    supports window-based replacement, and writes both JSON project files and a
    flat CSV export for downstream inspection.

Example:
    project = TrackLabelProject(source_path="/path/to/real.npy")
    project.replace_tracks_in_window(start_sample=0, end_sample=120000, tracks=[...])
    project.save_json("/path/to/manual_labels.json")
    project.export_csv("/path/to/manual_labels.csv")

Outputs:
    - JSON project file with source metadata, edit timestamp, and track points.
    - Flat CSV with one row per point for downstream scripts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from autotrack.core.track_extractor_graph import Track, TrackPoint


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class EditableTrackPoint:
    ch_idx: int
    t_idx: int
    time_s: float
    offset_m: float
    amp: float = 0.0
    score: float = 1.0
    source: str = "manual"


@dataclass
class EditableTrackRecord:
    track_id: int
    direction: str
    points: list[EditableTrackPoint] = field(default_factory=list)
    total_score: float = 0.0
    mean_speed_kmh: float = float("nan")
    source: str = "manual"
    note: str = ""


class TrackLabelProject:
    def __init__(
        self,
        *,
        source_path: str = "",
        source_kind: str = "",
        fs_hz: float = 1000.0,
        dx_m: float = 100.0,
    ) -> None:
        self.version = 1
        self.source_path = str(source_path)
        self.source_kind = str(source_kind)
        self.fs_hz = float(fs_hz)
        self.dx_m = float(dx_m)
        self.created_utc = _utc_now_iso()
        self.updated_utc = self.created_utc
        self.tracks: list[EditableTrackRecord] = []

    @classmethod
    def from_json(cls, json_path: str | Path) -> "TrackLabelProject":
        path = Path(json_path).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
        project = cls(
            source_path=str(payload.get("source_path", "")),
            source_kind=str(payload.get("source_kind", "")),
            fs_hz=float(payload.get("fs_hz", 1000.0)),
            dx_m=float(payload.get("dx_m", 100.0)),
        )
        project.version = int(payload.get("version", 1))
        project.created_utc = str(payload.get("created_utc", project.created_utc))
        project.updated_utc = str(payload.get("updated_utc", project.updated_utc))
        project.tracks = []
        for item in payload.get("tracks", []):
            points = [
                EditableTrackPoint(
                    ch_idx=int(p["ch_idx"]),
                    t_idx=int(p["t_idx"]),
                    time_s=float(p["time_s"]),
                    offset_m=float(p["offset_m"]),
                    amp=float(p.get("amp", 0.0)),
                    score=float(p.get("score", 1.0)),
                    source=str(p.get("source", item.get("source", "manual"))),
                )
                for p in item.get("points", [])
            ]
            project.tracks.append(
                EditableTrackRecord(
                    track_id=int(item["track_id"]),
                    direction=str(item.get("direction", "forward")),
                    points=project._sorted_points(points),
                    total_score=float(item.get("total_score", 0.0)),
                    mean_speed_kmh=float(item.get("mean_speed_kmh", float("nan"))),
                    source=str(item.get("source", "manual")),
                    note=str(item.get("note", "")),
                )
            )
        project.recompute_all()
        project.updated_utc = str(payload.get("updated_utc", project.updated_utc))
        return project

    @staticmethod
    def _sorted_points(points: list[EditableTrackPoint]) -> list[EditableTrackPoint]:
        dedup: dict[int, EditableTrackPoint] = {}
        for point in points:
            current = dedup.get(int(point.ch_idx))
            if current is None or float(point.score) >= float(current.score):
                dedup[int(point.ch_idx)] = point
        return sorted(dedup.values(), key=lambda p: (int(p.ch_idx), int(p.t_idx)))

    @staticmethod
    def _point_speed_kmh(p1: EditableTrackPoint, p2: EditableTrackPoint) -> float:
        dt = abs(float(p2.time_s) - float(p1.time_s))
        dx = abs(float(p2.offset_m) - float(p1.offset_m))
        if dt <= 1e-9 or dx <= 1e-9:
            return float("nan")
        return float(3.6 * dx / dt)

    @classmethod
    def _mean_speed_kmh(cls, points: list[EditableTrackPoint]) -> float:
        if len(points) < 2:
            return float("nan")
        speeds = []
        for idx in range(len(points) - 1):
            speed = cls._point_speed_kmh(points[idx], points[idx + 1])
            if np.isfinite(speed):
                speeds.append(float(speed))
        return float(np.mean(speeds)) if speeds else float("nan")

    def touch(self) -> None:
        self.updated_utc = _utc_now_iso()

    def _next_track_id(self) -> int:
        if not self.tracks:
            return 0
        return max(int(track.track_id) for track in self.tracks) + 1

    def renumber_tracks(self) -> None:
        ordered = sorted(
            self.tracks,
            key=lambda tr: min((int(p.t_idx) for p in tr.points), default=10**15),
        )
        for idx, track in enumerate(ordered):
            track.track_id = int(idx)
        self.tracks = ordered

    def recompute_track(self, track: EditableTrackRecord) -> None:
        track.points = self._sorted_points(track.points)
        track.total_score = float(np.sum([float(p.score) for p in track.points])) if track.points else 0.0
        track.mean_speed_kmh = self._mean_speed_kmh(track.points)

    def recompute_all(self) -> None:
        for track in self.tracks:
            self.recompute_track(track)
        self.renumber_tracks()
        self.touch()

    def load_source_defaults(self, *, source_path: str, source_kind: str, fs_hz: float, dx_m: float) -> None:
        self.source_path = str(source_path)
        self.source_kind = str(source_kind)
        self.fs_hz = float(fs_hz)
        self.dx_m = float(dx_m)
        self.touch()

    def visible_tracks(self, start_sample: int, end_sample: int) -> list[EditableTrackRecord]:
        visible: list[EditableTrackRecord] = []
        for track in self.tracks:
            if any(int(start_sample) <= int(point.t_idx) < int(end_sample) for point in track.points):
                visible.append(track)
        return visible

    def remove_tracks_in_window(self, start_sample: int, end_sample: int) -> int:
        start = int(start_sample)
        end = int(end_sample)
        kept: list[EditableTrackRecord] = []
        removed = 0
        for track in self.tracks:
            new_points = [point for point in track.points if not (start <= int(point.t_idx) < end)]
            if len(new_points) != len(track.points):
                removed += 1
            if new_points:
                track.points = new_points
                self.recompute_track(track)
                kept.append(track)
        self.tracks = kept
        self.renumber_tracks()
        self.touch()
        return removed

    @classmethod
    def from_backend_tracks(
        cls,
        tracks: list[Track],
        *,
        source: str = "auto",
    ) -> list[EditableTrackRecord]:
        out: list[EditableTrackRecord] = []
        for idx, track in enumerate(tracks):
            points = [
                EditableTrackPoint(
                    ch_idx=int(point.ch_idx),
                    t_idx=int(point.t_idx),
                    time_s=float(point.time_s),
                    offset_m=float(point.offset_m),
                    amp=float(point.amp),
                    score=float(point.score),
                    source=str(source),
                )
                for point in track.points
            ]
            out.append(
                EditableTrackRecord(
                    track_id=int(idx),
                    direction=str(track.direction),
                    points=points,
                    total_score=float(track.total_score),
                    mean_speed_kmh=float(track.mean_speed_kmh),
                    source=str(source),
                )
            )
        return out

    def replace_tracks_in_window(
        self,
        *,
        start_sample: int,
        end_sample: int,
        tracks: list[EditableTrackRecord],
    ) -> None:
        self.remove_tracks_in_window(start_sample, end_sample)
        next_id = self._next_track_id()
        for track in tracks:
            copied = EditableTrackRecord(
                track_id=int(next_id),
                direction=str(track.direction),
                points=[EditableTrackPoint(**asdict(point)) for point in track.points],
                total_score=float(track.total_score),
                mean_speed_kmh=float(track.mean_speed_kmh),
                source=str(track.source),
                note=str(track.note),
            )
            self.recompute_track(copied)
            self.tracks.append(copied)
            next_id += 1
        self.renumber_tracks()
        self.touch()

    def add_track(self, direction: str, source: str = "manual") -> EditableTrackRecord:
        track = EditableTrackRecord(track_id=self._next_track_id(), direction=str(direction), source=str(source))
        self.tracks.append(track)
        self.touch()
        return track

    def get_track(self, track_id: int) -> Optional[EditableTrackRecord]:
        for track in self.tracks:
            if int(track.track_id) == int(track_id):
                return track
        return None

    def delete_track(self, track_id: int) -> bool:
        before = len(self.tracks)
        self.tracks = [track for track in self.tracks if int(track.track_id) != int(track_id)]
        changed = len(self.tracks) != before
        if changed:
            self.renumber_tracks()
            self.touch()
        return changed

    def upsert_point(
        self,
        *,
        track_id: int,
        ch_idx: int,
        t_idx: int,
        offset_m: float,
        amp: float,
        score: float = 1.0,
        source: str = "manual",
    ) -> None:
        track = self.get_track(track_id)
        if track is None:
            raise KeyError(f"Track {track_id} not found")
        point = EditableTrackPoint(
            ch_idx=int(ch_idx),
            t_idx=int(t_idx),
            time_s=float(t_idx) / max(1e-9, float(self.fs_hz)),
            offset_m=float(offset_m),
            amp=float(amp),
            score=float(score),
            source=str(source),
        )
        replaced = False
        for idx, old in enumerate(track.points):
            if int(old.ch_idx) == int(point.ch_idx):
                track.points[idx] = point
                replaced = True
                break
        if not replaced:
            track.points.append(point)
        track.source = "manual"
        self.recompute_track(track)
        self.touch()

    def remove_point(self, *, track_id: int, ch_idx: int) -> bool:
        track = self.get_track(track_id)
        if track is None:
            return False
        before = len(track.points)
        track.points = [point for point in track.points if int(point.ch_idx) != int(ch_idx)]
        changed = len(track.points) != before
        if not changed:
            return False
        if track.points:
            track.source = "manual"
            self.recompute_track(track)
        else:
            self.delete_track(track_id)
        self.touch()
        return True

    def nearest_point(
        self,
        *,
        start_sample: int,
        end_sample: int,
        x_m: float,
        t_idx: int,
        x_tol_m: float,
        t_tol_samples: int,
        only_track_id: Optional[int] = None,
    ) -> Optional[tuple[EditableTrackRecord, EditableTrackPoint]]:
        best: Optional[tuple[EditableTrackRecord, EditableTrackPoint]] = None
        best_cost = float("inf")
        for track in self.visible_tracks(start_sample, end_sample):
            if only_track_id is not None and int(track.track_id) != int(only_track_id):
                continue
            for point in track.points:
                if not (int(start_sample) <= int(point.t_idx) < int(end_sample)):
                    continue
                dx = abs(float(point.offset_m) - float(x_m))
                dt = abs(int(point.t_idx) - int(t_idx))
                if dx > float(x_tol_m) or dt > int(t_tol_samples):
                    continue
                cost = (dx / max(1e-6, float(x_tol_m))) ** 2 + (dt / max(1, int(t_tol_samples))) ** 2
                if cost < best_cost:
                    best_cost = cost
                    best = (track, point)
        return best

    def to_payload(self) -> dict:
        return {
            "version": int(self.version),
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "fs_hz": float(self.fs_hz),
            "dx_m": float(self.dx_m),
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "track_count": int(len(self.tracks)),
            "tracks": [
                {
                    "track_id": int(track.track_id),
                    "direction": str(track.direction),
                    "total_score": float(track.total_score),
                    "mean_speed_kmh": float(track.mean_speed_kmh),
                    "source": str(track.source),
                    "note": str(track.note),
                    "points": [asdict(point) for point in track.points],
                }
                for track in self.tracks
            ],
        }

    @staticmethod
    def track_payload(track: EditableTrackRecord) -> dict:
        return {
            "track_id": int(track.track_id),
            "direction": str(track.direction),
            "total_score": float(track.total_score),
            "mean_speed_kmh": float(track.mean_speed_kmh),
            "source": str(track.source),
            "note": str(track.note),
            "points": [asdict(point) for point in track.points],
        }

    def save_json(self, json_path: str | Path) -> str:
        path = Path(json_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.touch()
        path.write_text(json.dumps(self.to_payload(), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def export_csv(self, csv_path: str | Path) -> str:
        path = Path(csv_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "track_id",
                    "direction",
                    "source",
                    "ch_idx",
                    "offset_m",
                    "t_idx",
                    "time_s",
                    "amp",
                    "score",
                    "mean_speed_kmh",
                ]
            )
            for track in sorted(self.tracks, key=lambda item: int(item.track_id)):
                for point in track.points:
                    writer.writerow(
                        [
                            int(track.track_id),
                            str(track.direction),
                            str(track.source),
                            int(point.ch_idx),
                            float(point.offset_m),
                            int(point.t_idx),
                            float(point.time_s),
                            float(point.amp),
                            float(point.score),
                            float(track.mean_speed_kmh) if np.isfinite(track.mean_speed_kmh) else np.nan,
                        ]
                    )
        return str(path)
