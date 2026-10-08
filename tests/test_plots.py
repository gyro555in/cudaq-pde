"""``benchmarks/plots.py`` draws the paper figures from stored JSON only.

CI has no stored benchmark outputs (they are gitignored), so the tests build small JSON
files with the same schema in a tmp dir; when the real stored files exist (a developer
checkout) they are drawn as well.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def plots():
    spec = importlib.util.spec_from_file_location(
        "plots", ROOT / "benchmarks" / "plots.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ synthetic inputs


def shot_study_json() -> dict:
    shots = [100, 1000, 10000, 100000]
    summary = [
        {
            "shots": s,
            "final_infidelity_mean": 1.6 / s,
            "final_infidelity_std": 0.5 / s,
            "final_l2_mean": 1.3 / np.sqrt(s),
            "norm_err_vs_fd_mean": 0.07 / np.sqrt(s),
        }
        for s in shots
    ]
    runs = {
        str(s): [{"infidelity": [0.0, 1.5 / s]}, {"infidelity": [0.0, 1.7 / s]}]
        for s in shots
    }
    params = {"n": 3, "depth": 2, "steps": 5, "seeds": 2, "shots": shots}
    return {"metadata": {}, "params": params, "summary": summary, "runs": runs}


def dynamic_range_json() -> dict:
    rows = []
    for nu, shots, res, resolved in [
        (0.5, [3e4, 6e4, 1e5], [True, True, True], True),
        (0.1, [3e3, 3e5, 3e8], [True, False, False], True),
        (0.02, [5e5, 3e11, 1e12], [False, False, False], False),
    ]:
        rows.append(
            {
                "case": "sin",
                "n": 6,
                "T": 1.0,
                "nu": nu,
                "eps": 0.05,
                "noise_floor": 1e-3,
                "probes": [8, 16, 24],
                "shots_per_probe": shots,
                "resolvable_on_hardware": res,
                "resolved": resolved,
            }
        )
    rows.append({**rows[0], "case": "asym"})  # other cases must be ignored
    return {"metadata": {}, "eps": 0.05, "dynamic_range": rows}


def spectrum_json() -> dict:
    shots = [100, 1000, 10000, 100000, 1000000]
    rms = [0.09 / np.sqrt(s / 100) for s in shots]
    rows = [
        {
            "n": 3,
            "shots": shots,
            "rms_error": [x * 1.01 for x in rms],
            "slope": -0.51,
            "predicted_rms_at_1e4": 0.0089,
        }
    ]
    rows += [
        {
            "n": n,
            "shots": shots,
            "rms_error": rms,
            "slope": -0.509,
            "predicted_rms_at_1e4": 0.0089,
        }
        for n in (4, 5, 6)
    ]
    return {"metadata": {}, "rows": rows}


def write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def check_outputs(png: Path, pdf: Path) -> None:
    assert png.read_bytes()[:8] == PNG_MAGIC and png.stat().st_size > 5_000
    assert pdf.read_bytes()[:5] == b"%PDF-" and pdf.stat().st_size > 2_000


# ------------------------------------------------------------------ figure builders


@pytest.mark.slow
def test_figure2_axes_guides_and_files(plots, tmp_path: Path) -> None:
    data = shot_study_json()
    fig = plots.build_figure2(data)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert "shots per circuit" in ax.get_xlabel() and "counts" in ax.get_xlabel()
    assert "error" in ax.get_ylabel()
    guides = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert len(guides) == 2
    slopes = sorted(
        np.polyfit(np.log(ln.get_xdata()), np.log(ln.get_ydata()), 1)[0]
        for ln in guides
    )
    assert slopes == pytest.approx([-1.0, -0.5], abs=1e-9)  # 1/S and 1/sqrt(S)
    assert len(ax.get_legend().get_texts()) == 3
    png, pdf = plots.figure2(
        write(tmp_path, "shot_study_1/shot_study.json", data), tmp_path / "f"
    )
    check_outputs(png, pdf)


@pytest.mark.slow
def test_figure3_selects_the_case_and_marks_the_resolvable_region(
    plots, tmp_path: Path
) -> None:
    data = dynamic_range_json()
    fig = plots.build_figure3(data)
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert "1/\\nu" in ax.get_xlabel() and "dimensionless" in ax.get_xlabel()
    assert "shots" in ax.get_ylabel() and "counts" in ax.get_ylabel()
    lines = [ln for ln in ax.get_lines() if ln.get_linestyle() == "-"]
    assert len(lines) == 3  # one line per probe
    for ln in lines:
        assert list(ln.get_xdata()) == pytest.approx(
            [2.0, 10.0, 50.0]
        )  # 1/nu, sin only
    markers = [ln for ln in ax.get_lines() if ln.get_linestyle() == "None"]
    filled = [
        ln
        for ln in markers
        if ln.get_marker() in "o^s" and ln.get_markerfacecolor() != "white"
    ]
    open_ = [ln for ln in ax.get_lines() if ln.get_markerfacecolor() == "white"]
    assert sum(len(ln.get_xdata()) for ln in filled) == 4  # resolvable points
    assert sum(len(ln.get_xdata()) for ln in open_) == 5  # unresolvable points
    crosses = [ln for ln in ax.get_lines() if ln.get_marker() == "x"]
    assert sum(len(ln.get_xdata()) for ln in crosses) == 3  # nu = 0.02: grid unresolved
    assert "assumption" in ax.get_title()
    path = write(tmp_path, "burgers_dynamic_range.json", data)
    check_outputs(*plots.figure3(path, tmp_path / "f"))
    with pytest.raises(ValueError):
        plots.build_figure3(data, case="sin", n=5)


@pytest.mark.slow
def test_figure4_groups_identical_curves_and_draws_the_prediction(
    plots, tmp_path: Path
) -> None:
    data = spectrum_json()
    groups = plots.group_equal_curves(data["rows"])
    assert [[r["n"] for r in g] for g in groups] == [[3], [4, 5, 6]]
    fig = plots.build_figure4(data)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert "shots" in ax.get_xlabel() and "RMS" in ax.get_ylabel()
    pred = [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]
    assert len(pred) == 1
    slope = np.polyfit(np.log(pred[0].get_xdata()), np.log(pred[0].get_ydata()), 1)[0]
    assert slope == pytest.approx(-0.5, abs=1e-9)
    check_outputs(
        *plots.figure4(write(tmp_path, "spectrum_shots.json", data), tmp_path / "f")
    )


def test_four_distinct_curves_are_refused(plots) -> None:
    shots = [100, 1000]
    rows = [
        {
            "n": n,
            "shots": shots,
            "rms_error": [0.1 * n, 0.03 * n],
            "slope": -0.5,
            "predicted_rms_at_1e4": 0.01,
        }
        for n in (3, 4, 5, 6)
    ]
    with pytest.raises(ValueError, match="distinct curves"):
        plots.build_figure4({"rows": rows})


def test_find_shot_study_prefers_job_directories(plots, tmp_path: Path) -> None:
    for name in ("shot_study_local", "shot_study_600001", "shot_study_606305"):
        write(tmp_path, f"{name}/shot_study.json", shot_study_json())
    assert plots.find_shot_study(tmp_path).parent.name == "shot_study_606305"
    only_local = tmp_path / "x"
    write(only_local, "shot_study_local/shot_study.json", shot_study_json())
    assert plots.find_shot_study(only_local).parent.name == "shot_study_local"
    with pytest.raises(FileNotFoundError):
        plots.find_shot_study(tmp_path / "empty")


@pytest.mark.slow
def test_command_line_skips_missing_inputs(
    plots, tmp_path: Path, capsys, monkeypatch
) -> None:
    write(tmp_path, "spectrum_shots.json", spectrum_json())
    monkeypatch.chdir(tmp_path)  # no results/ here: figure 2 has no input
    monkeypatch.setattr(
        "sys.argv",
        [
            "plots.py",
            "--outdir",
            str(tmp_path / "out"),
            "--spectrum",
            str(tmp_path / "spectrum_shots.json"),
            "--dynamic-range",
            str(tmp_path / "none.json"),
        ],
    )
    plots.main()
    out = capsys.readouterr().out
    assert "figure 2: skipped" in out and "figure 3: skipped" in out
    assert (
        "figure 4: " in out
        and (tmp_path / "out" / "figure4_power_spectrum.png").exists()
    )


# ------------------------------------------------------- the real stored files


@pytest.mark.slow
def test_stored_json_figures_when_present(plots, tmp_path: Path) -> None:
    drawn = 0
    try:
        check_outputs(*plots.figure2(None, tmp_path))
        drawn += 1
    except FileNotFoundError:
        pass
    for fn, rel in [
        (plots.figure3, "benchmarks/output/burgers_dynamic_range.json"),
        (plots.figure4, "benchmarks/output/spectrum_shots.json"),
    ]:
        if (ROOT / rel).exists():
            check_outputs(*fn(ROOT / rel, tmp_path))
            drawn += 1
    if not drawn:
        pytest.skip("no stored benchmark JSON in this checkout")
