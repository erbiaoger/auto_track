from __future__ import annotations

from pathlib import Path
from unittest import mock

from autotrack.dl.validate_single_vehicle_realbg_exact import main as validate_realbg_exact_main


def test_validate_single_vehicle_realbg_exact_wrapper_invokes_all_steps(tmp_path: Path) -> None:
    with mock.patch("autotrack.dl.validate_single_vehicle_realbg_exact.build_realistic_benchmark_main", return_value=0) as build_main, \
        mock.patch("autotrack.dl.validate_single_vehicle_realbg_exact.evaluate_single_vehicle_main", return_value=0) as eval_main, \
        mock.patch("autotrack.dl.validate_single_vehicle_realbg_exact.plot_single_vehicle_benchmark_compare_main", return_value=0) as plot_main:
        rc = validate_realbg_exact_main(
            [
                "--model",
                str(tmp_path / "model.pt"),
                "--out-dir",
                str(tmp_path / "validate_realbg_exact"),
                "--samples",
                "4",
                "--device",
                "cpu",
            ]
        )
    assert rc == 0
    assert build_main.called
    assert eval_main.called
    assert plot_main.called
