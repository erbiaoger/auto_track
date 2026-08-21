from __future__ import annotations

from pathlib import Path
from unittest import mock

from autotrack.dl.train_single_vehicle_exact import main as train_exact_main
from autotrack.dl.validate_single_vehicle_exact import main as validate_exact_main


def test_train_single_vehicle_exact_wrapper_invokes_benchmark_and_train(tmp_path: Path) -> None:
    with mock.patch("autotrack.dl.train_single_vehicle_exact.build_exact_benchmark_main", return_value=0) as build_main, \
        mock.patch("autotrack.dl.train_single_vehicle_exact.train_single_vehicle_main", return_value=0) as train_main:
        rc = train_exact_main(
            [
                "--out-dir",
                str(tmp_path / "train_exact"),
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
    assert build_main.called
    assert train_main.called


def test_validate_single_vehicle_exact_wrapper_invokes_all_steps(tmp_path: Path) -> None:
    with mock.patch("autotrack.dl.validate_single_vehicle_exact.build_exact_benchmark_main", return_value=0) as build_main, \
        mock.patch("autotrack.dl.validate_single_vehicle_exact.train_single_vehicle_exact_main", return_value=0) as train_main, \
        mock.patch("autotrack.dl.validate_single_vehicle_exact.evaluate_single_vehicle_main", return_value=0) as eval_main, \
        mock.patch("autotrack.dl.validate_single_vehicle_exact.plot_single_vehicle_benchmark_compare_main", return_value=0) as plot_main:
        rc = validate_exact_main(
            [
                "--out-dir",
                str(tmp_path / "validate_exact"),
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
    assert build_main.called
    assert train_main.called
    assert eval_main.called
    assert plot_main.called
