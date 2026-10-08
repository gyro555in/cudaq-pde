"""``benchmarks/check_provenance.py`` on synthetic metadata files."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "check_provenance",
    Path(__file__).resolve().parent.parent / "benchmarks" / "check_provenance.py",
)
cp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cp)

HEAD = "a" * 40
OLD = "b" * 40


def meta(commit=HEAD, dirty=False, target="qpp-cpu", gpu=None, job=None):
    return {
        "git_commit": commit, "git_dirty": dirty, "target": target,
        "gpu_name": gpu, "slurm_job_id": job,
    }  # fmt: skip


def put(root: Path, rel: str, m: dict, wrap=True) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": m} if wrap else m))


def run(root, *args, capsys=None):
    return cp.main(["--root", str(root), "--head", HEAD, *args])


def test_clean_tree_passes_and_table_lists_columns(tmp_path, capsys):
    put(tmp_path, "benchmarks/output/a.json", meta())
    put(tmp_path, "results/shot_study_12/shot_study.json", meta(job="12"))
    assert run(tmp_path) == 0
    out = capsys.readouterr().out
    for col in ("file", "commit", "git_dirty", "target", "GPU", "job id"):
        assert col in out
    assert "aaaaaaa" in out and "12" in out


def test_dirty_file_fails(tmp_path):
    put(tmp_path, "benchmarks/output/a.json", meta(dirty=True))
    assert run(tmp_path) == 1


def test_dirty_flag_missing_fails(tmp_path):
    put(tmp_path, "benchmarks/output/a.json", {"git_commit": HEAD})
    assert run(tmp_path) == 1


def test_other_commit_fails_unless_allowed(tmp_path):
    put(tmp_path, "benchmarks/output/a.json", meta(commit=OLD))
    assert run(tmp_path) == 1
    assert run(tmp_path, "--allow-commit-mismatch") == 0


def test_allow_mismatch_still_fails_on_dirty(tmp_path):
    put(tmp_path, "benchmarks/output/a.json", meta(commit=OLD, dirty=True))
    assert run(tmp_path, "--allow-commit-mismatch") == 1


def test_gpu_file_is_checked_against_commit_option(tmp_path):
    put(tmp_path, "benchmarks/output/g.json", meta(OLD, target="nvidia", gpu="A100"))
    put(tmp_path, "benchmarks/output/c.json", meta())
    assert run(tmp_path) == 1
    assert run(tmp_path, "--commit", OLD) == 0
    assert run(tmp_path, "--commit", OLD[:9]) == 0  # abbreviated sha


def test_commit_option_does_not_excuse_cpu_files(tmp_path):
    put(tmp_path, "benchmarks/output/c.json", meta(commit=OLD))
    assert run(tmp_path, "--commit", OLD) == 1


def test_expect_glob_for_the_shot_study(tmp_path):
    put(tmp_path, "results/shot_study_9/shot_study.json", meta(commit=OLD))
    assert run(tmp_path) == 1
    assert run(tmp_path, "--expect", f"results/shot_study_9/*={OLD[:7]}") == 0


def test_flat_log_metadata_is_read(tmp_path):
    put(tmp_path, "logs/gpu_tests_5_metadata.json", meta(job="5"), wrap=False)
    assert run(tmp_path) == 0
    put(tmp_path, "logs/gpu_tests_6_metadata.json", meta(dirty=True), wrap=False)
    assert run(tmp_path) == 1


def test_documented_but_missing_output_fails(tmp_path):
    put(tmp_path, "benchmarks/output/a.json", meta())
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/reproduce_paper.md").write_text(
        "| T | `cmd` | `s.py` | `benchmarks/output/b.json`, "
        "`benchmarks/output/a.json` |"
    )
    assert run(tmp_path) == 1


def test_unreadable_json_fails_and_scratch_shot_study_ignored(tmp_path):
    put(tmp_path, "results/shot_study_local/shot_study.json", meta(commit=OLD))
    put(tmp_path, "benchmarks/output/a.json", meta())
    assert run(tmp_path) == 0
    (tmp_path / "benchmarks/output/bad.json").write_text("{not json")
    assert run(tmp_path) == 1


def test_no_files_is_an_error(tmp_path):
    assert run(tmp_path) == 1


def test_bad_expect_argument(tmp_path):
    assert run(tmp_path, "--expect", "nosha") == 2


@pytest.mark.parametrize(
    "sha,want,same",
    [
        ("abcdef1234", "abcdef1", True),
        ("abcdef1234", "abcdef2", False),
        ("abcdef1234", "abc", False),
    ],
)
def test_same_requires_seven_characters(sha, want, same):
    assert cp._same(sha, want) is same


def test_gpu_release_copies_are_read(tmp_path):
    put(tmp_path, "results/gpu_release_7/mid_circuit_nvidia.json", meta(job="7"))
    assert run(tmp_path) == 0
    put(tmp_path, "results/gpu_release_8/x.json", meta(dirty=True, job="8"))
    assert run(tmp_path) == 1
