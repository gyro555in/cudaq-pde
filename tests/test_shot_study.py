"""The shot-study script runs end to end at a tiny size (qpp-cpu)."""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path

import cudaq
import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    yield
    cudaq.reset_target()


def load():
    spec = importlib.util.spec_from_file_location("shot_study", BENCH / "shot_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_runs_and_reports_the_accounting(tmp_path: Path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("cudaq.get_state called by the shot study")

    study = load()
    # preprocessing (the simulator fit) is allowed to read states; patch afterwards
    prep = study.prepare(3, 2, 2)
    monkeypatch.setattr(study, "prepare", lambda *a: prep)
    monkeypatch.setattr(cudaq, "get_state", forbidden)
    out = study.main(
        [
            "--n",
            "3",
            "--steps",
            "2",
            "--shots",
            "200",
            "--seeds",
            "2",
            "--iterations",
            "3",
            "--outdir",
            str(tmp_path),
        ]
    )
    row = out["summary"][0]
    P = 3 * 3
    assert row["circuits_per_step"] == 3 * 2 * P + 1
    assert row["shots_per_step"] == 3 * 2 * P * 200 + 200
    assert row["seeds"] == 2
    saved = json.loads((tmp_path / "shot_study.json").read_text())
    assert saved["metadata"]["git_commit"] and "runs" in saved
    for key in (
        "norm_err_vs_fd_mean",
        "norm_err_statistical_mean",
        "norm_err_state_mean",
    ):
        assert row[key] >= 0.0


def test_sbatch_is_syntactically_valid_and_targets_cpu() -> None:
    text = (BENCH / "shot_study.sbatch").read_text()
    assert "--partition=cpu-genoa" in text and "--output=logs/%x_%j.out" in text
    assert "gpus" not in text  # a CPU job
