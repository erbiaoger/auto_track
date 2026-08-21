from __future__ import annotations

from pathlib import Path
from unittest import mock

from autotrack.dl.run_single_vehicle_exact import main as run_exact_main


def test_run_single_vehicle_exact_wrapper_invokes_validate(tmp_path: Path) -> None:
    with mock.patch("autotrack.dl.run_single_vehicle_exact.validate_exact_main", return_value=0) as validate_main:
        rc = run_exact_main(
            [
                "--out-dir",
                str(tmp_path / "run_exact"),
                "--samples",
                "4",
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--hidden-dim",
                "16",
                "--device",
                "cpu",
            ]
        )
    assert rc == 0
    assert validate_main.called
