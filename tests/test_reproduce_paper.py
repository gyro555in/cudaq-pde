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
# helper modules that record no metadata and write no output of their own
NOT_A_STEP = {"mid_circuit_probe_kernels.py"}


def rows() -> list[dict]:
    out = []
    for line in DOC.splitlines():
        if (
            not line.startswith("| ")
            or line.startswith("| Paper item")
            or "---|" in line
        ):
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


def test_the_table_is_not_empty_and_covers_the_named_items() -> None:
    items = " ".join(r["item"] for r in ROWS)
    for name in ("Table 1", "Table 3", "Table 7", "Figure 2", "Figure 3", "Figure 4"):
        assert name in items, name
    assert len(ROWS) >= 15


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
        longest = max(pieces(path.stem), key=len)
        assert longest and longest in source, (
            f"{output}: {longest!r} not in {row['scripts']}"
        )
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
