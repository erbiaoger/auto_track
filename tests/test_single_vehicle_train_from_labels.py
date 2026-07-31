import tempfile
import unittest
from pathlib import Path

import numpy as np

from autotrack.dl.train_single_vehicle_from_labels import main as train_from_labels_main
from autotrack.labeling.track_label_project import EditableTrackPoint, EditableTrackRecord, TrackLabelProject


class SingleVehicleTrainFromLabelsTest(unittest.TestCase):
    def test_finetunes_from_label_project_in_one_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "source.npy"
            data = np.zeros((4000, 51), dtype=np.float32)
            data[1000:1020, 5] = 2.0
            np.save(source, data)

            project = TrackLabelProject(source_path=str(source), source_kind="real_npy", fs_hz=100.0, dx_m=10.0)
            project.tracks = [
                EditableTrackRecord(
                    track_id=0,
                    direction="forward",
                    points=[
                        EditableTrackPoint(ch_idx=5, t_idx=1000, time_s=10.0, offset_m=50.0, amp=2.0, score=1.0),
                        EditableTrackPoint(ch_idx=6, t_idx=1045, time_s=10.45, offset_m=60.0, amp=2.0, score=1.0),
                        EditableTrackPoint(ch_idx=7, t_idx=1090, time_s=10.9, offset_m=70.0, amp=2.0, score=1.0),
                    ],
                    total_score=3.0,
                    mean_speed_kmh=80.0,
                )
            ]
            labels_path = tmp / "manual_labels.json"
            project.save_json(labels_path)

            out_dir = tmp / "out"

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "train_single_vehicle_from_labels",
                    "--out-dir",
                    str(out_dir),
                    "--source-npy",
                    str(source),
                    "--labels-json",
                    str(labels_path),
                    "--window-seconds",
                    "5",
                    "--margin-seconds",
                    "1",
                    "--windows-per-track",
                    "3",
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--log-every",
                    "0",
                ]
                self.assertEqual(train_from_labels_main(), 0)
            finally:
                sys.argv = old_argv

            self.assertTrue((out_dir / "checkpoint_best.pt").is_file())
            self.assertTrue((out_dir / "train_history.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
