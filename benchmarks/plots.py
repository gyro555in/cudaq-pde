"""Paper figures from the STORED benchmark JSON (no simulation is run here).

    python benchmarks/plots.py                      # all figures that have their input
    python benchmarks/plots.py --outdir results/figures

Figure 2  shot noise of the measurable variational solver, from
          ``results/shot_study_<jobid>/shot_study.json`` (final infidelity, relative L2
          error and norm error against shots per circuit, log-log, 1/S and 1/sqrt(S)
          guides). Written by ``benchmarks/shot_study.sbatch``.
Figure 3  cost of the pointwise Burgers velocity against 1/nu at the three probes
          (n = 6, case sin, T = 1), from
          ``benchmarks/output/burgers_dynamic_range.json`` (written by
          ``benchmarks/burgers_dynamic_range.py``). Filled markers: the probe is above
          the hardware noise floor; open markers: unresolvable regardless of shots.
Figure 4  RMS error of the sampled power spectrum against shots with the multinomial
          prediction, from ``benchmarks/output/spectrum_shots.json``
          (``benchmarks/spectrum_shots.py``).

Each figure is written as PNG (200 dpi) and PDF (vector) to ``results/figures/``.
matplotlib only. Colours are the first three slots of the validated categorical palette
(blue, orange, aqua: worst all-pairs colour-vision Delta E 9.2, normal vision 24), and
every series also has its own marker shape, so colour is never the only cue. Guides and
predictions are grey and dashed.
"""

from __future__ import annotations

import argparse
import json
import re
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
GUIDE = "#6b6b6b"
SERIES = [BLUE, ORANGE, AQUA]
MARKERS = ["o", "s", "^"]
FIGSIZE = (6.4, 4.3)
DEFAULT_OUTDIR = Path("results/figures")


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _new_figure() -> tuple[Figure, object]:
    fig = Figure(figsize=FIGSIZE, dpi=200, facecolor="white")
    ax = fig.add_subplot(111)
    _style(ax)
    return fig, ax


def save(fig: Figure, outdir: Path, stem: str) -> tuple[Path, Path]:
    """Write ``stem.png`` and ``stem.pdf`` to ``outdir`` and return both paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png, pdf = outdir / f"{stem}.png", outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return png, pdf


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ Figure 2


def find_shot_study(root: str | Path = "results") -> Path:
    """The ``shot_study_<jobid>/shot_study.json`` with the highest job id.

    Directories whose suffix is a job number win over ``shot_study_local``.
    """
    cands = sorted(Path(root).glob("shot_study_*/shot_study.json"))
    numbered = [
        (int(m.group(1)), p)
        for p in cands
        if (m := re.search(r"_(\d+)$", p.parent.name))
    ]
    if numbered:
        return max(numbered)[1]
    if cands:
        return cands[-1]
    raise FileNotFoundError(f"no shot_study_*/shot_study.json under {root}")


def build_figure2(data: dict) -> Figure:
    """Final error against shots per circuit, log-log, with 1/S and 1/sqrt(S) guides."""
    rows = data["summary"]
    shots = np.array([r["shots"] for r in rows], dtype=float)
    infid = np.array([r["final_infidelity_mean"] for r in rows])
    infid_std = np.array([r["final_infidelity_std"] for r in rows])
    l2 = np.array([r["final_l2_mean"] for r in rows])
    norm = np.array([r["norm_err_vs_fd_mean"] for r in rows])
    p = data["params"]

    fig, ax = _new_figure()
    # individual seeds as light dots behind the mean
    for s, rs in data["runs"].items():
        ax.plot(
            [float(s)] * len(rs),
            [r["infidelity"][-1] for r in rs],
            ls="none",
            marker="o",
            ms=3,
            mfc=BLUE,
            mec="none",
            alpha=0.25,
            zorder=2,
        )
    ax.errorbar(
        shots,
        infid,
        yerr=np.minimum(infid_std, infid * 0.999),
        color=BLUE,
        marker="o",
        ms=6,
        lw=2,
        capsize=3,
        zorder=4,
        label="final infidelity to FD-Euler (mean, 1 sd over seeds)",
    )
    ax.plot(
        shots,
        l2,
        color=ORANGE,
        marker="s",
        ms=6,
        lw=2,
        zorder=4,
        label="final relative L2 error (mean)",
    )
    ax.plot(
        shots,
        norm,
        color=AQUA,
        marker="^",
        ms=6,
        lw=2,
        zorder=4,
        label="final norm error vs FD (mean)",
    )

    ref = np.array([shots[0], shots[-1]])
    ax.plot(ref, infid[0] * (shots[0] / ref), ls="--", color=GUIDE, lw=1.2, zorder=3)
    ax.plot(
        ref, l2[0] * (shots[0] / ref) ** 0.5, ls="--", color=GUIDE, lw=1.2, zorder=3
    )
    ax.annotate(
        r"$\propto 1/S$",
        (ref[1], infid[0] * shots[0] / ref[1]),
        (5, -12),
        textcoords="offset points",
        color=GUIDE,
        fontsize=9,
    )
    ax.annotate(
        r"$\propto 1/\sqrt{S}$",
        (ref[1], l2[0] * (shots[0] / ref[1]) ** 0.5),
        (5, 4),
        textcoords="offset points",
        color=GUIDE,
        fontsize=9,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shots per circuit, S (counts)")
    ax.set_ylabel("error (dimensionless)")
    ax.set_title(
        f"Shot noise of the measurable variational step (n = {p['n']}, "
        f"depth {p['depth']}, "
        f"{p['steps']} steps, {p['seeds']} seeds)",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def figure2(path: str | Path | None = None, outdir: Path = DEFAULT_OUTDIR):
    """Figure 2 from a stored shot-study JSON (default: latest job in ``results/``)."""
    data = load(path if path is not None else find_shot_study())
    return save(build_figure2(data), outdir, "figure2_shot_study")


# ------------------------------------------------------------------ Figure 3


def _pi_label(probe: int, n: int) -> str:
    frac = Fraction(2 * probe, 2**n).limit_denominator(16)  # x / pi
    num, den = frac.numerator, frac.denominator
    num_s = "" if num == 1 else str(num)
    return rf"$x = {num_s}\pi/{den}$" if den != 1 else rf"$x = {num_s}\pi$"


def build_figure3(data: dict, case: str = "sin", n: int = 6, T: float = 1.0) -> Figure:
    """Shots for a standard error of the velocity against 1/nu at the three probes."""
    rows = sorted(
        (
            r
            for r in data["dynamic_range"]
            if r["case"] == case and r["n"] == n and r["T"] == T
        ),
        key=lambda r: r["nu"],
        reverse=True,
    )
    if not rows:
        raise ValueError(f"no rows for case={case}, n={n}, T={T}")
    inv_nu = np.array([1.0 / r["nu"] for r in rows])
    probes = rows[0]["probes"]
    floor, eps = rows[0]["noise_floor"], rows[0]["eps"]

    fig, ax = _new_figure()
    for k, probe in enumerate(probes):
        y = np.array([r["shots_per_probe"][k] for r in rows])
        ok = np.array([r["resolvable_on_hardware"][k] for r in rows])
        bad_grid = np.array([not r["resolved"] for r in rows])
        color, marker = SERIES[k], MARKERS[k]
        ax.plot(inv_nu, y, color=color, lw=2, zorder=3)
        ax.plot(
            inv_nu[ok],
            y[ok],
            ls="none",
            marker=marker,
            ms=8,
            mfc=color,
            mec=color,
            zorder=4,
        )
        ax.plot(
            inv_nu[~ok],
            y[~ok],
            ls="none",
            marker=marker,
            ms=8,
            mfc="white",
            mec=color,
            mew=1.8,
            zorder=4,
        )
        ax.plot(
            inv_nu[bad_grid],
            y[bad_grid],
            ls="none",
            marker="x",
            ms=11,
            mec=INK,
            mew=1.2,
            zorder=5,
        )
        ax.annotate(
            _pi_label(probe, n),
            (inv_nu[-1], y[-1]),
            (7, -3),
            textcoords="offset points",
            color=INK,
            fontsize=9,
        )
    handles = [
        Line2D(
            [],
            [],
            color=SERIES[k],
            marker=MARKERS[k],
            ms=7,
            lw=2,
            label=_pi_label(p, n),
        )
        for k, p in enumerate(probes)
    ] + [
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=8,
            mfc=GUIDE,
            mec=GUIDE,
            label=f"filled: min(p$_{{j\\pm1}}$) $\\geq$ {floor:g} (resolvable)",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            ms=8,
            mfc="white",
            mec=GUIDE,
            mew=1.8,
            label="open: below the floor (unresolvable for any S)",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="x",
            ms=10,
            mec=INK,
            label="cross: grid solution unresolved (aliased)",
        ),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    ax.set_yscale("log")
    ax.set_xlabel(r"inverse viscosity, $1/\nu$ (dimensionless)")
    ax.set_ylabel(rf"total shots for $\sigma_u = {eps:g}$ (counts)")
    ax.set_title(
        f"Pointwise Burgers velocity: cost against $1/\\nu$ "
        f"({case} $x$, $n = {n}$, $T = {T:g}$)\n"
        f"hardware noise floor {floor:g} is a modelling assumption",
        fontsize=9,
    )
    ax.set_xlim(inv_nu.min() * 0.85, inv_nu.max() * 1.25)
    fig.tight_layout()
    return fig


def figure3(
    path: str | Path = "benchmarks/output/burgers_dynamic_range.json",
    outdir: Path = DEFAULT_OUTDIR,
    case: str = "sin",
    n: int = 6,
    T: float = 1.0,
):
    """Figure 3 from the stored dynamic-range JSON."""
    return save(build_figure3(load(path), case, n, T), outdir, "figure3_dynamic_range")


# ------------------------------------------------------------------ Figure 4


def group_equal_curves(rows: list[dict], rtol: float = 1e-4) -> list[list[dict]]:
    """Consecutive ``n`` whose RMS curves agree within ``rtol`` share one line."""
    groups: list[list[dict]] = []
    for r in sorted(rows, key=lambda r: r["n"]):
        if groups and np.allclose(
            groups[-1][0]["rms_error"], r["rms_error"], rtol=rtol, atol=0
        ):
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def build_figure4(data: dict) -> Figure:
    """Measured RMS error of the sampled power spectrum vs the multinomial law."""
    rows = data["rows"]
    groups = group_equal_curves(rows)
    if len(groups) > len(SERIES):
        raise ValueError(
            f"{len(groups)} distinct curves; at most {len(SERIES)} are drawn"
        )
    fig, ax = _new_figure()
    for k, grp in enumerate(groups):
        r = grp[0]
        label = (
            f"$n = {grp[0]['n']}$"
            if len(grp) == 1
            else f"$n = {grp[0]['n']}$ to ${grp[-1]['n']}$"
        )
        slopes = [g["slope"] for g in grp]
        label += f" (log-log slope {np.mean(slopes):.2f})"
        ax.plot(
            r["shots"],
            r["rms_error"],
            color=SERIES[k],
            marker=MARKERS[k],
            ms=6,
            lw=2,
            label=label,
            zorder=4,
        )
    # multinomial prediction sqrt((1 - sum p^2)/S): stored at S = 1e4, scaled as S^-1/2
    shots = np.array(rows[0]["shots"], dtype=float)
    pred = rows[0]["predicted_rms_at_1e4"] * np.sqrt(1e4 / shots)
    ax.plot(
        shots,
        pred,
        ls="--",
        color=GUIDE,
        lw=1.4,
        zorder=3,
        label=r"multinomial prediction $\sqrt{(1-\sum p^2)/S}$ ($n = 3$)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shots, S (counts)")
    ax.set_ylabel("RMS L2 error of the power spectrum $P_k$ (dimensionless)")
    ax.set_title("Sampled power spectrum: error against shots", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    return fig


def figure4(
    path: str | Path = "benchmarks/output/spectrum_shots.json",
    outdir: Path = DEFAULT_OUTDIR,
):
    """Figure 4 from the stored spectrum-shots JSON."""
    return save(build_figure4(load(path)), outdir, "figure4_power_spectrum")


# ------------------------------------------------------------------ command line


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--shot-study", type=Path, default=None)
    ap.add_argument(
        "--dynamic-range",
        type=Path,
        default=Path("benchmarks/output/burgers_dynamic_range.json"),
    )
    ap.add_argument(
        "--spectrum", type=Path, default=Path("benchmarks/output/spectrum_shots.json")
    )
    a = ap.parse_args()
    jobs = [
        ("figure 2", lambda: figure2(a.shot_study, a.outdir)),
        ("figure 3", lambda: figure3(a.dynamic_range, a.outdir)),
        ("figure 4", lambda: figure4(a.spectrum, a.outdir)),
    ]
    for name, job in jobs:
        try:
            png, pdf = job()
            print(f"{name}: {png}  {pdf}")
        except FileNotFoundError as exc:
            print(f"{name}: skipped, missing input ({exc})")


if __name__ == "__main__":
    main()
