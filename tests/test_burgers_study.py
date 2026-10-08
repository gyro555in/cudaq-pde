"""Dynamic-range study rows, the sampled confirmation, and the phi-spectrum readout."""

import numpy as np
import pytest

from cudaq_pde import burgers as B
from cudaq_pde import classical
from cudaq_pde.solvers import burgers_estimator as be
from cudaq_pde.solvers import burgers_study as st

L = 2 * np.pi


@pytest.fixture(autouse=True)
def cpu_target():
    import cudaq

    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


# ------------------------------------------------------------------ dynamic range


def test_probes_avoid_the_shock_centre_and_wrap_nowhere() -> None:
    for n in (5, 6):
        pr = st.probe_indices(n)
        assert (pr * L / 2**n).tolist() == pytest.approx(
            [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        )
        assert 2**n // 2 not in pr


def test_exact_probabilities_agree_with_the_grid_where_resolved() -> None:
    n, nu, T = 6, 0.5, 1.0
    d = B.discrete_burgers(np.sin(classical.grid(n)), nu, T)
    p = st.true_probabilities(np.sin, nu, T, n)
    assert np.max(np.abs(np.log(d.p) - np.log(p))) < 1e-9


def test_valley_depth_follows_exp_minus_two_over_nu() -> None:
    """The initial ``p`` valley of sin x is ``exp(-2/nu)``."""
    for nu in (0.5, 0.1, 0.05):
        p0 = np.exp((np.cos(classical.grid(8)) - 1) / nu)
        assert p0.min() / p0.max() == pytest.approx(np.exp(-2 / nu), rel=1e-6)


def test_resolved_flag_and_the_fp64_floor() -> None:
    easy = st.study_row("sin", 0.5, 5)
    assert easy["resolved"] and not easy["below_fp64_floor"]
    hard = st.study_row("sin", 0.02, 5)
    assert (
        hard["below_fp64_floor"] and not hard["resolved"] and hard["n_unresolved"] > 0
    )
    assert hard["min_p_grid"] > 1e-20 > hard["min_p_true"]  # roundoff, not physics


def test_cost_grows_as_nu_shrinks_at_the_hardest_probe() -> None:
    worst = [st.study_row("sin", nu, 5)["shots_worst"] for nu in (0.5, 0.1, 0.05)]
    assert worst[0] < worst[1] < worst[2]
    assert worst[1] / worst[0] > 1e3  # ~ exp(2/nu) growth


def test_error_terms_scale_as_expected() -> None:
    r5, r6 = st.study_row("sin", 0.5, 5), st.study_row("sin", 0.5, 6)
    for a, b in zip(r5["err_fd"], r6["err_fd"], strict=True):
        assert 3.5 < a / b < 4.5  # O(h^2) finite difference
    assert max(r6["err_grid"]) < 1e-6  # nu = 0.5: well resolved


def test_shots_column_is_the_estimator_formula() -> None:
    n, nu = 5, 0.1
    r = st.study_row("asym", nu, n, eps=0.07)
    d = B.discrete_burgers(st.CASES["asym"](classical.grid(n)), nu, 1.0)
    qp, qm = be.probe_probabilities(d.p, d.p_succ, st.probe_indices(n))
    np.testing.assert_allclose(
        r["shots_per_probe"], be.shots_for_error(qp, qm, 0.07, nu, L / 2**n)
    )
    assert r["shots_best"] <= r["shots_worst"] <= r["shots_to_see_all"] * 10


@pytest.mark.slow
@pytest.mark.parametrize("nu,n,probe", [(0.5, 5, 8), (0.5, 5, 4), (0.1, 5, 4)])
def test_sampled_scatter_matches_the_prediction(nu: float, n: int, probe: int) -> None:
    r = st.study_row("sin", nu, n)
    shots = int(min(max(r["shots_per_probe"]), 3e6))
    out = st.sampled_confirmation("sin", nu, n, 1.0, probe, shots, seeds=8)
    assert out["max_abs_z"] < 4.0
    assert 0.3 < out["rms_error"] / out["predicted_sigma"] < 1.8


# ------------------------------------------------------------------ spectrum


@pytest.mark.parametrize("nu", [0.5, 0.1, 0.05, 0.02])
def test_phi_spectrum_error_follows_the_multinomial_prediction(nu: float) -> None:
    r = st.spectrum_row("asym", nu, 5, 1.0, shots=40_000, seed=3)
    assert 0.6 < r["rms_error"] / r["rms_predicted"] < 1.6
    assert r["tv_distance"] < 0.05


def test_spectrum_readout_cost_does_not_blow_up_with_small_nu() -> None:
    errs = [
        st.spectrum_row("sin", nu, 5, 1.0, shots=40_000, seed=1)["rms_error"]
        for nu in (0.5, 0.05, 0.02)
    ]
    assert max(errs) < 0.02  # no ln of tiny numbers, unlike the pointwise velocity


def test_weighted_mean_square_velocity_identity() -> None:
    """``4 nu^2 sum k^2 P_k`` is the p-weighted mean of ``u^2`` (not the energy)."""
    n, nu, T = 6, 0.5, 1.0
    d = B.discrete_burgers(np.sin(classical.grid(n)), nu, T)
    m = np.fft.fftfreq(2**n, 1.0 / 2**n)
    from cudaq_pde.solvers.spectral_measure import classical_power_spectrum

    msq = 4 * nu**2 * np.sum(m**2 * classical_power_spectrum(d.phiT))
    assert msq == pytest.approx(np.sum(d.p * d.u_spectral**2), rel=1e-6)
    energy = np.mean(d.u_spectral**2)
    assert abs(msq - energy) > 1e-3 * energy  # a different weighting, labelled as such


def test_sampled_weighted_velocity_is_within_its_standard_error() -> None:
    n, nu, T, shots = 5, 0.5, 1.0, 100_000
    r = st.spectrum_row("sin", nu, n, T, shots=shots, seed=7)
    d = B.discrete_burgers(np.sin(classical.grid(n)), nu, T)
    from cudaq_pde.solvers.spectral_measure import classical_power_spectrum

    m = np.fft.fftfreq(2**n, 1.0 / 2**n)
    P = classical_power_spectrum(d.phiT)
    k2 = m**2
    var = (np.sum(k2**2 * P) - np.sum(k2 * P) ** 2) / r["accepted"]
    se = 4 * nu**2 * np.sqrt(var)
    assert (
        abs(r["weighted_msq_velocity_hat"] - r["weighted_msq_velocity_classical"])
        < 4 * se
    )
