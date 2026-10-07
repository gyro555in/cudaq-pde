"""The gpu and slow markers: skipped by default, enabled by their own option."""

from pathlib import Path

import pytest

CONFTEST = (Path(__file__).parent / "conftest.py").read_text()
MARKERS_INI = """
[pytest]
markers =
    gpu: needs a GPU
    slow: takes more than about 1 s
"""
TESTS = """
import pytest

def test_fast(): pass

@pytest.mark.slow
def test_slow(): pass

@pytest.mark.gpu
def test_gpu(): pass

@pytest.mark.slow
@pytest.mark.gpu
def test_both(): pass
"""


@pytest.fixture
def project(pytester: pytest.Pytester) -> pytest.Pytester:
    """A throwaway project using the repository's real conftest.py."""
    pytester.makeini(MARKERS_INI)
    pytester.makeconftest(CONFTEST.replace('pytest_plugins = ["pytester"]', ""))
    pytester.makepyfile(TESTS)
    return pytester


def outcomes(pytester: pytest.Pytester, *args: str) -> dict[str, int]:
    return pytester.runpytest_subprocess(
        "-p", "no:cacheprovider", *args
    ).parseoutcomes()


def test_default_runs_only_the_fast_test(project) -> None:
    assert outcomes(project) == {"passed": 1, "skipped": 3}


def test_run_slow_enables_slow_but_not_gpu(project) -> None:
    assert outcomes(project, "--run-slow") == {"passed": 2, "skipped": 2}


def test_run_gpu_enables_gpu_but_not_slow(project) -> None:
    assert outcomes(project, "--run-gpu") == {"passed": 2, "skipped": 2}


def test_both_options_run_everything(project) -> None:
    assert outcomes(project, "--run-slow", "--run-gpu") == {"passed": 4}


def test_skip_reasons_name_the_option(project) -> None:
    result = project.runpytest_subprocess("-p", "no:cacheprovider", "-rs")
    result.stdout.fnmatch_lines(["*needs --run-slow*", "*needs --run-gpu*"])


SHARD_TESTS = "\n".join(f"def test_{i}(): pass" for i in range(7))


def test_shards_partition_the_tests(pytester: pytest.Pytester) -> None:
    pytester.makeini(MARKERS_INI)
    pytester.makeconftest(CONFTEST.replace('pytest_plugins = ["pytester"]', ""))
    pytester.makepyfile(SHARD_TESTS)
    passed = [
        outcomes(pytester, "--shard", f"{k}/3").get("passed", 0) for k in (1, 2, 3)
    ]
    assert sorted(passed) == [2, 2, 3] and sum(passed) == 7
    names = [
        pytester.runpytest_subprocess(
            "-p", "no:cacheprovider", "--shard", f"{k}/3", "--collect-only", "-q"
        ).outlines
        for k in (1, 2, 3)
    ]
    seen = [ln for out in names for ln in out if "::test_" in ln]
    assert len(seen) == 7 and len(set(seen)) == 7  # disjoint and complete


def test_bad_shard_spec_is_a_usage_error(pytester: pytest.Pytester) -> None:
    pytester.makeini(MARKERS_INI)
    pytester.makeconftest(CONFTEST.replace('pytest_plugins = ["pytester"]', ""))
    pytester.makepyfile(SHARD_TESTS)
    for bad in ("0/3", "4/3", "x", "1/0"):
        res = pytester.runpytest_subprocess("-p", "no:cacheprovider", "--shard", bad)
        assert res.ret != 0
