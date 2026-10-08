"""The fp64 agreement check used by ``benchmarks/mid_circuit_probe.sbatch`` (GPU job).

The job is never run here; the same code runs on ``qpp-cpu`` (fp64) so the check, the
printed log lines and the JSON with metadata are exercised on every CI run.
"""

import importlib.util
import json
import subprocess
from pathlib import Path

import cudaq
import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "oneshot_fp64_agreement", BENCH / "oneshot_fp64_agreement.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_cli_runs_on_the_cpu_target_and_writes_json(
    mod, tmp_path, monkeypatch, capsys
) -> None:
    out = tmp_path / "out" / "oneshot_fp64_agreement.json"
    monkeypatch.setattr("sys.argv", ["x", "--target", "qpp-cpu", "--out", str(out)])
    assert mod.main() == 0
    cudaq.reset_target()
    log = capsys.readouterr().out
    assert "fp64 state-vector agreement of the one-shot routes" in log
    assert log.count(": max error") == 12  # n = 3, 4 x 2 conventions x 3 routes
    assert "worst:" in log and "PASS" in log
    data = json.loads(out.read_text())
    assert data["passed"] is True and data["worst"] < 1e-12
    assert len(data["records"]) == 12
    assert {(r["n"], r["nyquist"], r["route"]) for r in data["records"]} == {
        (n, q, r) for n in (3, 4) for q in ("zero", "negative") for r in "ABC"
    }
    assert data["metadata"]["cudaq_version"] and "git_dirty" in data["metadata"]
    assert data["metadata"]["target"] == "qpp-cpu"


def test_sbatch_calls_the_script_and_is_valid_shell() -> None:
    text = (BENCH / "mid_circuit_probe.sbatch").read_text()
    assert "benchmarks/oneshot_fp64_agreement.py --target nvidia --option fp64" in text
    assert "<<'PY'" not in text  # the inline heredoc is gone
    assert (
        subprocess.run(
            ["bash", "-n", str(BENCH / "mid_circuit_probe.sbatch")]
        ).returncode
        == 0
    )
