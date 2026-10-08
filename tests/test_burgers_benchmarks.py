"""The Burgers benchmark scripts run end to end and write their JSON."""

import json
import runpy
from pathlib import Path

import cudaq
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def cpu_target():
    yield
    cudaq.reset_target()


@pytest.mark.slow
@pytest.mark.parametrize(
    "script,keys",
    [
        ("burgers_resources.py", {"circuits", "cost", "metadata"}),
        (
            "burgers_dynamic_range.py",
            {"dynamic_range", "sampled", "spectrum", "metadata"},
        ),
    ],
)
def test_script_runs_and_writes_json(
    script, keys, tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(ROOT / "benchmarks" / script), run_name="__main__")
    out = capsys.readouterr().out
    assert "Table 1" in out
    data = json.loads(
        (tmp_path / "benchmarks/output" / script.replace(".py", ".json")).read_text()
    )
    assert keys <= set(data)
    assert data["metadata"]["cudaq_version"]
