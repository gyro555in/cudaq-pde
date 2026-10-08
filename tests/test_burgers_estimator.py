"""Velocity estimator: delta-method variance and bias against exact multinomial Monte
Carlo, zero counts, probe bookkeeping, and a sampled CUDA-Q check (slow)."""

import numpy as np
import pytest

from cudaq_pde import burgers as B
from cudaq_pde import classical
from cudaq_pde.solvers import burgers_estimator as be

NU, H = 0.1, 2 * np.pi / 32
QP, QM = 0.006, 0.002  # joint probabilities of the two neighbours (3 : 1)
R = 200_000
TRUE = -(NU / (2 * H)) * np.log(QP / QM)


def draw(lam_minus: float, seed: int = 0):
    S = int(round(lam_minus / QM))
    c = np.random.default_rng(seed).multinomial(S, [QP, QM, 1 - QP - QM], size=R)
    return S, c[:, 0], c[:, 1]


def plain(cp, cm):
    return -(NU / (2 * H)) * (np.log(cp) - np.log(cm))


def haldane(cp, cm):
    return -(NU / (2 * H)) * (np.log(cp + 0.5) - np.log(cm + 0.5))


@pytest.mark.parametrize("lam", [30, 100, 1000])
def test_delta_variance_and_bias_match_monte_carlo(lam: int) -> None:
    S, cp, cm = draw(lam)
    u = plain(cp, cm)
    sd = np.sqrt(be.velocity_variance(QP, QM, S, NU, H))
    ratio = u.var() / sd**2
    assert abs(ratio - 1.0) < 4 * np.sqrt(2 / R) + 3.0 / lam  # O(1/lambda) correction
    bias_units = (u.mean() - TRUE) / sd
    pred_units = be.velocity_bias(QP, QM, S, NU, H) / sd
    assert abs(bias_units - pred_units) < 4 / np.sqrt(R) + 0.1 * abs(pred_units)


@pytest.mark.parametrize("lam", [1, 3])
def test_probability_of_an_undefined_estimate(lam: int) -> None:
    S, cp, cm = draw(lam)
    p_zero = 1 - (1 - np.exp(-QP * S)) * (1 - np.exp(-QM * S))
    obs = np.mean((cp == 0) | (cm == 0))
    assert abs(obs - p_zero) < 4 * np.sqrt(p_zero * (1 - p_zero) / R) + 0.01


def test_delta_method_fails_at_one_expected_count_and_the_mask_says_so() -> None:
    S, cp, cm = draw(1)
    ok = (cp > 0) & (cm > 0)
    sd2 = float(be.velocity_variance(QP, QM, S, NU, H))
    assert abs(plain(cp[ok], cm[ok]).var() / sd2 - 1.0) > 0.4  # not usable
    counts = np.zeros(8, dtype=int)
    counts[[1, 3]] = [cp[0], cm[0]]
    est = be.velocity_estimate(counts, [2], NU, 2 * np.pi, c_min=30)
    assert not est.valid[0]


@pytest.mark.parametrize("lam", [10, 30, 100])
def test_haldane_removes_the_leading_bias(lam: int) -> None:
    S, cp, cm = draw(lam)
    sd = np.sqrt(be.velocity_variance(QP, QM, S, NU, H))
    plain_bias = abs((plain(cp, cm).mean() - TRUE) / sd) if lam > 10 else 0.09
    hal_bias = abs((haldane(cp, cm).mean() - TRUE) / sd)
    assert hal_bias < 0.25 * plain_bias + 4 / np.sqrt(R)


def test_shots_for_error_inverts_the_variance() -> None:
    qp, qm = np.array([0.01, 1e-4]), np.array([0.02, 1e-3])
    S = be.shots_for_error(qp, qm, 0.05, NU, H)
    np.testing.assert_allclose(be.velocity_variance(qp, qm, S, NU, H), 0.05**2)


# ------------------------------------------------------------------ bookkeeping


def test_estimate_from_counts_is_exact_when_counts_follow_p() -> None:
    d = B.discrete_burgers(
        np.sin(classical.grid(5)) + 0.5 * np.sin(2 * classical.grid(5)), 0.5, 1.0
    )
    counts = np.round(d.p * 1e9).astype(np.int64)  # S q = 1e9 p: ratios exact to 1e-9
    est = be.velocity_estimate(counts, np.arange(32), 0.5)
    np.testing.assert_allclose(est.u_plain, d.u_fd, atol=1e-6)
    assert est.valid.all() and est.n_undefined == 0


def test_probes_wrap_around_and_zero_counts_are_explicit() -> None:
    counts = np.array([50, 0, 40, 60, 0, 0, 7, 100])
    probes = [0, 3, 4, 7]
    # neighbours (j+1, j-1): probe 0 -> (1, 7) = (0, 100), probe 3 -> (4, 2) = (0, 40),
    # probe 4 -> (5, 3) = (0, 60), probe 7 -> (0, 6) = (50, 7): wraps around
    est = be.velocity_estimate(counts, probes, 0.5, c_min=10)
    assert np.isnan(est.u_plain[:3]).all() and np.isinf(est.sigma_plugin[:3]).all()
    assert est.n_undefined == 3
    assert np.isfinite(est.u_haldane).all()  # (c + 1/2) is defined everywhere
    assert not est.valid.any()  # zeros and the count 7 are below c_min
    assert est.c_plus[3] == 50 and est.c_minus[3] == 7
    h = 2 * np.pi / 8
    assert est.u_plain[3] == pytest.approx(-(0.5 / (2 * h)) * np.log(50 / 7))


def test_inverted_ratio_gives_the_negative_velocity() -> None:
    counts = np.array([30, 80, 120, 60, 25, 70, 90, 40])
    a = be.velocity_estimate(counts, np.arange(8), 0.5)
    b = be.velocity_estimate(counts[::-1].copy(), np.arange(8), 0.5)
    np.testing.assert_allclose(b.u_plain, -a.u_plain[::-1])


# ------------------------------------------------------------------ sampled (CUDA-Q)


@pytest.mark.slow
def test_sampled_velocity_has_the_predicted_scatter() -> None:
    import cudaq

    from cudaq_pde.solvers import burgers_quantum as bq

    cudaq.set_target("qpp-cpu")
    n, nu, T, shots = 4, 0.5, 0.8, 200_000
    u0 = np.sin(classical.grid(n)) + 0.5 * np.sin(2 * classical.grid(n))
    d = B.discrete_burgers(u0, nu, T)
    probes = np.array([2, 4, 6])
    h = 2 * np.pi / 2**n
    qp, qm = be.probe_probabilities(d.p, d.p_succ, probes)
    sd = np.sqrt(be.velocity_variance(qp, qm, shots, nu, h))
    zs = []
    for seed in range(10):
        s = bq.sample_heat(d.phi0, nu, T, 2 * np.pi, "A", shots, seed=100 + seed)
        est = be.velocity_estimate(s.grid_counts, probes, nu)
        assert est.valid.all()
        zs.append((est.u_plain - d.u_fd[probes]) / sd)
        np.testing.assert_allclose(est.sigma_plugin, sd, rtol=0.1)
    z = np.array(zs)
    assert np.max(np.abs(z)) < 4.0
    assert 0.6 < np.sqrt(np.mean(z**2)) < 1.5  # unit scatter within 30 samples
    cudaq.reset_target()
