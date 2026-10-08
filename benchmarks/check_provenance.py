"""Check the run metadata of every stored paper output.

Reads the metadata block (the run metadata, under the key ``metadata`` or as the
whole file for ``logs/*_metadata.json``) of

* ``benchmarks/output/*.json``
* ``results/shot_study_<jobid>/shot_study.json`` (Slurm job directories only)
* ``results/gpu_release_<jobid>/*.json`` (copies made by ``gpu_release.sbatch``)
* ``logs/*_metadata.json``

plus every ``benchmarks/output/*.json`` that ``docs/reproduce_paper.md`` names without a
placeholder (a missing one is reported as MISSING), and prints one row per file: file,
commit, git_dirty, target, GPU, job id. Exit code 1 if any file is dirty, missing,
unreadable, or was written at a commit other than the expected one.

Expected commit: HEAD, except

* files that ran on a GPU use ``--commit SHA`` when given (GPU jobs are submitted
  separately, from the release commit);
* ``--expect GLOB=SHA`` (repeatable) sets the commit for files matching GLOB, e.g. the
  shot study that stays as a record at its own commit;
* ``--allow-commit-mismatch`` turns every commit mismatch into a warning (a dirty tree
  still fails).

    python benchmarks/check_provenance.py
    python benchmarks/check_provenance.py --commit <sha> --allow-commit-mismatch
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = (
    "benchmarks/output/*.json",
    "results/shot_study_*/shot_study.json",
    "results/gpu_release_*/*.json",
    "logs/*_metadata.json",
)
DOC = "docs/reproduce_paper.md"
COLUMNS = ("file", "commit", "git_dirty", "target", "GPU", "job id", "status")


def git_head(root: Path) -> str | None:
    """Full SHA of HEAD in ``root`` or None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root, capture_output=True, text=True, check=True,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def documented_outputs(root: Path) -> list[str]:
    """``benchmarks/output/<name>.json`` paths in the doc without a placeholder."""
    doc = root / DOC
    if not doc.exists():
        return []
    found = re.findall(r"`(benchmarks/output/[^`<>]+\.json)`", doc.read_text())
    return sorted(set(found))


def collect_files(root: Path) -> list[str]:
    """Relative paths of all stored outputs (glob patterns plus the documented ones)."""
    paths: set[str] = set(documented_outputs(root))
    for pat in PATTERNS:
        paths.update(str(p.relative_to(root)) for p in root.glob(pat))
    # shot studies from a Slurm job only (results/shot_study_local is a scratch run)
    paths = {
        p for p in paths
        if not p.startswith("results/shot_study_")
        or re.fullmatch(r"results/shot_study_\d+/shot_study\.json", p)
    }  # fmt: skip
    return sorted(paths)


def read_metadata(path: Path) -> dict | None:
    """The metadata block of a JSON file, or None if unreadable."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    meta = data.get("metadata", data)
    return meta if isinstance(meta, dict) else None


def _same(commit: str | None, expected: str) -> bool:
    """SHA equality that accepts an abbreviated ``expected`` (at least 7 characters)."""
    return bool(commit) and len(expected) >= 7 and commit.startswith(expected)


def expected_commit(
    rel: str, meta: dict, head: str, commit: str | None, expects: dict[str, str]
) -> str:
    """The commit this file must carry (see the module docstring)."""
    for glob, sha in expects.items():
        if fnmatch.fnmatch(rel, glob):
            return sha
    if commit and meta.get("gpu_name"):
        return commit
    return head


def check_file(
    rel: str,
    meta: dict | None,
    head: str,
    commit: str | None = None,
    expects: dict[str, str] | None = None,
    allow_mismatch: bool = False,
) -> tuple[tuple[str, ...], bool]:
    """One table row and whether the file passes."""
    if meta is None:
        return (rel, "-", "-", "-", "-", "-", "MISSING OR UNREADABLE"), False
    sha = meta.get("git_commit")
    dirty = meta.get("git_dirty")
    want = expected_commit(rel, meta, head, commit, expects or {})
    problems = []
    if dirty is not False:
        problems.append("DIRTY" if dirty else "git_dirty not recorded")
    if not _same(sha, want):
        if allow_mismatch:
            problems.append(f"commit differs from {want[:7]} (allowed)")
        else:
            problems.append(f"COMMIT MISMATCH (want {want[:7]})")
    # failures: dirty, unrecorded dirty flag, or a commit mismatch that is not allowed
    ok = not any(
        p in ("DIRTY", "git_dirty not recorded") or p.startswith("COMMIT MISMATCH")
        for p in problems
    )
    row = (
        rel,
        (sha or "-")[:7],
        str(dirty),
        str(meta.get("target") or "-"),
        str(meta.get("gpu_name") or "-"),
        str(meta.get("slurm_job_id") or "-"),
        "ok" if not problems else "; ".join(problems),
    )
    return row, ok


def format_table(rows: list[tuple[str, ...]]) -> str:
    """Fixed-width table with a header."""
    table = [COLUMNS, *rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(COLUMNS))]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip()
        for r in table
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path("."), help="repository root")
    ap.add_argument("--head", help="expected commit (default: git HEAD of --root)")
    ap.add_argument("--commit", help="expected commit of files that ran on a GPU")
    ap.add_argument(
        "--expect", action="append", default=[], metavar="GLOB=SHA",
        help="expected commit for files matching GLOB (repeatable)",
    )  # fmt: skip
    ap.add_argument(
        "--allow-commit-mismatch", action="store_true",
        help="report a commit mismatch as a warning (a dirty tree still fails)",
    )  # fmt: skip
    args = ap.parse_args(argv)

    root = args.root.resolve()
    head = args.head or git_head(root)
    if not head:
        print("cannot determine HEAD; pass --head", file=sys.stderr)
        return 2
    expects: dict[str, str] = {}
    for item in args.expect:
        glob, sep, sha = item.rpartition("=")
        if not sep or not glob or not sha:
            print(f"--expect needs GLOB=SHA, got {item!r}", file=sys.stderr)
            return 2
        expects[glob] = sha

    rows, failed = [], 0
    for rel in collect_files(root):
        meta = read_metadata(root / rel) if (root / rel).exists() else None
        row, ok = check_file(
            rel, meta, head, args.commit, expects, args.allow_commit_mismatch
        )
        rows.append(row)
        failed += not ok
    print(f"expected commit (HEAD): {head}")
    if args.commit:
        print(f"expected commit of GPU outputs: {args.commit}")
    print(format_table(rows))
    print(f"{len(rows)} files, {failed} failing")
    return 1 if failed or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
