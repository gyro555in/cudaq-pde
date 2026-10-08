"""``docs/reproduce_paper.md`` names only scripts and outputs that exist.

The stored outputs are gitignored, so "exists" means: the script is in the repository,
the command runs that script, and the output file name (up to the ``<placeholder>``
parts) is one the named script(s) write. Every benchmark that records run metadata
must appear in the map.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = (ROOT / "docs" / "reproduce_paper.md").read_text()
OUTPUT_ROOTS = ("benchmarks/output/", "results/", "logs/")
# not a step of the map: a helper module with no output of its own, and a benchmark that
# no paper table uses
NOT_A_STEP = {"mid_circuit_probe_kernels.py", "qiskit_roundtrip.py"}


def rows() -> list[dict]:
    """Rows of the four-column tables (header ``Paper item | Command | ...``)."""
    out = []
    in_table = False
    for line in DOC.splitlines():
        if line.startswith("| Paper item | Command"):
            in_table = True
            continue
        if not line.startswith("|"):
            in_table = False
            continue
        if not in_table or "---|" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        assert len(cells) == 4, line
        tick = [re.findall(r"`([^`]+)`", c) for c in cells]
        out.append(
            {
                "item": cells[0],
                "command": tick[1][0],
                "scripts": tick[2],
                "outputs": tick[3],
            }
        )
    return out


ROWS = rows()


def pieces(name: str) -> list[str]:
    """Literal parts of a file name around the ``<placeholder>`` parts."""
    return re.split(r"<[^>]+>", name)


def test_every_paper_table_and_figure_is_mapped_and_none_is_unassigned() -> None:
    items = [r["item"] for r in ROWS]
    for k in range(1, 9):
        assert any(re.match(rf"Table {k}\b", i) for i in items), f"Table {k}"
    for k in (2, 3, 4):
        assert any(
            re.match(rf"Figure {k}\b", i) or f"Figure {k} " in i for i in items
        ), k
    assert not [i for i in items if "unassigned" in i.lower()]
    assert "unassigned" not in DOC.lower()


def test_the_paper_numbers_point_at_the_agreed_sources() -> None:
    expected = {
        "Table 1": {"spectral_resources.py"},
        "Table 2": {"cost_circuit_comparison.py", "measurable_resources.py"},
        "Table 3": {"variational_report.py"},
        "Table 4": {"shot_study.py", "shot_study.sbatch"},
        "Table 5": {"oneshot_resources.py"},
        "Table 6": {"burgers_resources.py"},
        "Table 7": {
            "mid_circuit_probe.sbatch",
            "mid_circuit_probe.py",
            "mid_circuit_probe_stats.py",
            "oneshot_fp64_agreement.py",  # called by mid_circuit_probe.sbatch
        },
        "Table 8": {"burgers_dynamic_range.py"},
        "Figure 2": {"shot_study.py", "shot_study.sbatch"},
        "Figure 3": {"burgers_dynamic_range.py"},
        "Figure 4": {"spectrum_shots.py"},
    }
    for item, names in expected.items():
        got = {
            Path(s).name
            for r in ROWS
            if re.match(rf"{item}\b", r["item"]) or f"and {item} " in r["item"]
            for s in r["scripts"]
        }
        assert names <= got, (item, sorted(names - got))


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["item"][:50])
def test_scripts_exist_and_the_command_runs_one_of_them(row: dict) -> None:
    for script in row["scripts"]:
        assert (ROOT / script).is_file(), f"{script} does not exist"
    assert any(s in row["command"] for s in row["scripts"]), row["command"]
    first = row["command"].split()
    assert first[0] in {"taskset", "sbatch", "python"}, row["command"]


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["item"][:50])
def test_output_names_are_written_by_the_named_scripts(row: dict) -> None:
    source = "\n".join((ROOT / s).read_text() for s in row["scripts"])
    for output in row["outputs"]:
        assert output.startswith(OUTPUT_ROOTS), output
        path = Path(output)
        if output.startswith("logs/"):  # sbatch log: <job-name>_<jobid>.out
            job_name = re.search(r"--job-name=(\S+)", source)
            assert job_name, f"no --job-name in {row['scripts']}"
            assert pieces(path.name)[0].startswith(
                job_name.group(1).replace("+", "")
            ), output
            continue
        for piece in (p for p in pieces(path.stem) if p):
            assert piece in source, f"{output}: {piece!r} not in {row['scripts']}"
        directory = pieces(str(path.parent))[0]
        # some scripts build the directory as ``Path(__file__).parent / "output"``
        ok = directory in source or (
            directory == "benchmarks/output" and '"output"' in source
        )
        assert ok, f"{output}: {directory!r} not in {row['scripts']}"


def test_every_metadata_benchmark_is_in_the_map() -> None:
    listed = {Path(s).name for r in ROWS for s in r["scripts"]}
    missing = [
        p.name
        for p in sorted((ROOT / "benchmarks").glob("*.py"))
        if "collect_metadata" in p.read_text()
        and p.name not in listed
        and p.name not in NOT_A_STEP
    ]
    assert not missing, f"benchmarks missing from docs/reproduce_paper.md: {missing}"


def test_figure_stems_match_the_plotting_module() -> None:
    plots = (ROOT / "benchmarks" / "plots.py").read_text()
    figure_rows = [r for r in ROWS if r["scripts"] == ["benchmarks/plots.py"]]
    assert [r["item"][:8] for r in figure_rows] == ["Figure 2", "Figure 3", "Figure 4"]
    for row in figure_rows:
        for output in row["outputs"]:
            assert output.startswith("results/figures/")
            assert Path(output).stem in plots
