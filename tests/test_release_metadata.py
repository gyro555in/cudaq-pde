"""Version and metadata files agree, so a tag cannot ship inconsistent metadata."""

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
VERSION = PYPROJECT["project"]["version"]


def test_version_is_0_1_0() -> None:
    assert VERSION == "0.1.0"


def test_installed_package_version_matches_pyproject() -> None:
    assert version("cudaq-pde") == VERSION


def test_citation_version_matches() -> None:
    text = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(VERSION)}$", text, flags=re.MULTILINE)
    assert "cff-version: 1.2.0" in text
    assert "Ramakrishna" in text and "Kushal" in text
    assert "Helmholtz-Zentrum Dresden-Rossendorf" in text
    # fields that are not decided must not be invented
    assert not re.search(r"^(date-released|license|doi|orcid):", text, re.MULTILINE)


def test_changelog_top_entry_matches() -> None:
    headings = re.findall(
        r"^## \[(\d+\.\d+\.\d+)\]", (ROOT / "CHANGELOG.md").read_text(), re.MULTILINE
    )
    assert headings[0] == VERSION


def test_readme_is_declared_and_exists() -> None:
    assert PYPROJECT["project"]["readme"] == "README.md"
    assert (ROOT / "README.md").stat().st_size > 1000
    assert PYPROJECT["project"]["authors"] == [{"name": "Kushal Ramakrishna"}]


def test_no_license_is_claimed_before_one_is_chosen() -> None:
    assert "license" not in PYPROJECT["project"]
    assert not (ROOT / "LICENSE").exists()
