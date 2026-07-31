from __future__ import annotations

from pathlib import Path
from unittest import mock

from autotrack.dl.run_single_vehicle_realbg_exact import main as run_realbg_exact_main
from autotrack.dl.train_single_vehicle_realbg_exact import main as train_realbg_exact_main


def test_train_single_vehicle_realbg_exact_wrapper_invokes_benchmark_and_train(tmp_path: Path) -> None:
    with mock.patch("autotrack.dl.train_single_vehicle_realbg_exact.build_realistic_benchmark_main", return_value=0) as build_main, \
        mock.patch("autotrack.dl.train_single_vehicle_realbg_exact.train_single_vehicle_main", return_value=0) as train_main:
        rc = train_realbg_exact_main(
            [
                "--out-dir",
                str(tmp_path / "train_realbg_exact"),
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


def test_run_single_vehicle_realbg_exact_wrapper_invokes_train_eval_plot(tmp_path: Path) -> None:
    with mock.patch("autotrack.dl.run_single_vehicle_realbg_exact.train_single_vehicle_realbg_exact_main", return_value=0) as train_main, \
        mock.patch("autotrack.dl.run_single_vehicle_realbg_exact.evaluate_single_vehicle_main", return_value=0) as eval_main, \
        mock.patch("autotrack.dl.run_single_vehicle_realbg_exact.plot_single_vehicle_benchmark_compare_main", return_value=0) as plot_main:
        rc = run_realbg_exact_main(
            [
                "--out-dir",
                str(tmp_path / "run_realbg_exact"),
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
    assert train_main.called
    assert eval_main.called
    assert plot_main.called
