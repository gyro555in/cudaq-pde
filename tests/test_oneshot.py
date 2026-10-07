"""One-shot advection-diffusion: plan identities, numpy checks of the building blocks,
and the three routes against ``classical.evolve_spectral``.

STATEVECTOR CHECK (test only): the route tests post-select the ancillas on the state
vector via ``oneshot_sim``; the hardware-path modules never read a state vector.
"""

import numpy as np
import pytest
from scipy.stats import chi2

from cudaq_pde import classical, encoding
from cudaq_pde.solvers import oneshot, oneshot_sim, sampling

TOL = 1e-12
C, NU, L, T = 0.7, 0.05, 2 * np.pi, 0.8


@pytest.fixture(autouse=True)
def cpu_target():
    import cudaq

    cudaq.set_target("qpp-cpu")


def field(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=2**n)


# ------------------------------------------------------------------ plan identities


def test_sigma_and_exponent_are_consistent() -> None:
    for n in (3, 5, 8):
        N, h = 2**n, L / 2**n
        a = oneshot.diffusion_exponent(NU, T, L)
        sigma = oneshot.gaussian_width(n, NU, T, L)
        assert sigma == pytest.approx(np.sqrt(2 * NU * T) / h, rel=1e-15)
        # Fourier multiplier of a Gaussian of width sigma cells (in grid cells)
        # is exp(-2 pi^2 sigma^2 m^2 / N^2)
        assert a == pytest.approx(2 * np.pi**2 * sigma**2 / N**2, rel=1e-14)


def test_sigma_doubles_with_every_qubit_at_fixed_time() -> None:
    sig = [oneshot.gaussian_width(n, NU, T, L) for n in range(3, 9)]
    assert np.allclose(np.array(sig[1:]) / np.array(sig[:-1]), 2.0, rtol=1e-14)


def test_real_space_kernel_is_the_gaussian_of_width_sigma() -> None:
    n, nu, T_, L_ = 6, 0.5, 1.0, 2 * np.pi
    a = oneshot.diffusion_exponent(nu, T_, L_)
    sigma = oneshot.gaussian_width(n, nu, T_, L_)
    w = oneshot.gaussian_weights(n, a)
    s = np.arange(-(2 ** (n - 1)), 2 ** (n - 1))
    # periodic Gaussian: sum of the images s + k N
    gauss = sum(
        np.exp(-((s + k * 2**n) ** 2) / (2 * sigma**2)) for k in range(-4, 5)
    ) / (sigma * np.sqrt(2 * np.pi))
    assert np.max(np.abs(w[s % 2**n] - gauss)) < 1e-12


def test_response_includes_the_nyquist_mode() -> None:
    n = 4
    a = 0.01
    g = oneshot.response(n, a)
    assert g[2 ** (n - 1)] == pytest.approx(np.exp(-a * (2 ** (n - 1)) ** 2))
    assert oneshot.modes(n)[2 ** (n - 1)] == -(2 ** (n - 1))


# ----------------------------------------------------------- numpy building blocks


def apply_sequence(state: np.ndarray, seq: oneshot.RySequence) -> np.ndarray:
    """Apply the RY-then-CX list to a numpy state (qubit k = bit k of the index)."""
    state = state.astype(np.float64).copy()
    idx = np.arange(state.size)
    for tgt, alpha, ctl in zip(seq.tgt, seq.alpha, seq.ctl, strict=True):
        c, s = np.cos(alpha / 2), np.sin(alpha / 2)
        lo = idx[((idx >> tgt) & 1) == 0]
        hi = lo | (1 << tgt)
        a0, a1 = state[lo].copy(), state[hi].copy()
        state[lo], state[hi] = c * a0 - s * a1, s * a0 + c * a1
        if ctl >= 0:
            on = idx[((idx >> ctl) & 1) == 1]
            sel = on[((on >> tgt) & 1) == 0]
            partner = sel | (1 << tgt)
            state[sel], state[partner] = state[partner].copy(), state[sel].copy()
    return state


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_multiplexed_ry_applies_the_requested_angle_per_control_value(k: int) -> None:
    theta = np.random.default_rng(k).uniform(-3, 3, size=2**k)
    seq = oneshot.multiplexed_ry(theta, list(range(k)), k)
    assert seq.n_ry == 2**k
    assert seq.n_cx == (2**k if k else 0)
    for j in range(2**k):
        state = np.zeros(2 ** (k + 1))
        state[j] = 1.0
        out = apply_sequence(state, seq)
        expect = np.zeros_like(out)
        expect[j] = np.cos(theta[j] / 2)
        expect[j | (1 << k)] = np.sin(theta[j] / 2)
        assert np.max(np.abs(out - expect)) < 1e-13


@pytest.mark.parametrize("q", [1, 2, 3, 4, 5])
def test_prepare_real_builds_signed_amplitudes(q: int) -> None:
    v = np.random.default_rng(q).normal(size=2**q)
    seq = oneshot.prepare_real(v, list(range(q)))
    assert seq.n_ry == 2**q - 1
    assert seq.n_cx == 2**q - 2
    state = np.zeros(2**q)
    state[0] = 1.0
    out = apply_sequence(state, seq)
    assert np.max(np.abs(out - v / np.linalg.norm(v))) < 1e-13
    back = apply_sequence_inverse(out, seq)
    assert np.max(np.abs(back - state)) < 1e-13


def apply_sequence_inverse(state: np.ndarray, seq: oneshot.RySequence) -> np.ndarray:
    """Apply ``seq.reversed()`` as CX-then-RY (what ``cx_ry_sequence`` does)."""
    rev = seq.reversed()
    state = state.copy()
    for tgt, alpha, ctl in zip(rev.tgt, rev.alpha, rev.ctl, strict=True):
        if ctl >= 0:
            single = oneshot.RySequence([tgt], [0.0], [ctl])
            state = apply_sequence(state, single)
        state = apply_sequence(state, oneshot.RySequence([tgt], [alpha], [-1]))
    return state


# ------------------------------------------------------------------ route B factors


@pytest.mark.parametrize("n", range(2, 9))
def test_contractions_reproduce_the_response_and_are_contractions(n: int) -> None:
    a = 0.013
    fac = oneshot.contractions(n, a)
    singles, doubles = fac.n_controls()
    assert (singles, doubles) == (n, n * (n - 1) // 2)
    assert fac.count == n * (n + 1) // 2
    assert all(w > 0 for w in fac.weight)
    assert all(0.0 <= np.cos(t / 2) <= 1.0 for t in fac.theta)
    assert (
        np.max(np.abs(oneshot.contraction_values(n, a) - oneshot.response(n, a)))
        < 1e-14
    )


@pytest.mark.parametrize("n", [3, 4, 5])
def test_twos_complement_decomposition_is_not_block_encodable(n: int) -> None:
    """Negative control: with the raw two's-complement bits ``m**2`` has negative
    cross terms with the sign bit, so its factors ``exp(-a w)`` exceed 1."""
    weights = np.array([2.0**b for b in range(n - 1)] + [-(2.0 ** (n - 1))])
    cross = [2 * weights[b] * weights[n - 1] for b in range(n - 1)]  # b < sign bit
    assert all(w < 0 for w in cross)
    assert all(np.exp(-0.01 * w) > 1.0 for w in cross)
    # and the quadratic form is right: sum_b w_b f_b squared = m**2
    for j in range(2**n):
        f = np.array([(j >> b) & 1 for b in range(n)])
        assert (weights @ f) ** 2 == oneshot.modes(n)[j] ** 2


# ------------------------------------------------------------------ routes vs reference


def run_case(n: int, route: str, nyquist: str, nu: float = NU, c: float = C):
    u0 = field(n, seed=n)
    res = oneshot_sim.evolve_oneshot(u0, c, nu, L, T, route, nyquist)
    ref = classical.evolve_spectral(u0, T, c, nu, L, nyquist)
    return u0, res, ref


@pytest.mark.parametrize("nyquist", ["zero", "negative"])
@pytest.mark.parametrize("route", ["A", "B", "C"])
@pytest.mark.parametrize("n", [3, 4])
def test_route_matches_the_spectral_reference(n: int, route: str, nyquist: str) -> None:
    u0, res, ref = run_case(n, route, nyquist)
    assert np.max(np.abs(res.u - ref)) < TOL * max(1.0, np.linalg.norm(u0))


@pytest.mark.parametrize("route", ["A", "B", "C"])
@pytest.mark.parametrize("n", [3, 4])
def test_success_probability_is_the_norm_ratio(n: int, route: str) -> None:
    u0, res, ref = run_case(n, route, "zero")
    ratio = np.vdot(ref, ref).real / np.vdot(u0, u0).real
    a = oneshot.diffusion_exponent(NU, T, L)
    assert oneshot.success_probability(u0, a) == pytest.approx(ratio, abs=TOL)
    assert res.p_succ == pytest.approx(ratio / res.lam**2, abs=TOL)


@pytest.mark.parametrize("nu", [0.0, 0.01, 0.2])
@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_route_handles_the_limits_of_the_decay(route: str, nu: float) -> None:
    u0, res, ref = run_case(3, route, "zero", nu=nu)
    assert np.max(np.abs(res.u - ref)) < TOL * np.linalg.norm(u0)


@pytest.mark.slow
@pytest.mark.parametrize("nyquist", ["zero", "negative"])
@pytest.mark.parametrize("route", ["A", "C"])
@pytest.mark.parametrize("n", [5, 6])
def test_larger_n_routes_a_and_c(n: int, route: str, nyquist: str) -> None:
    u0, res, ref = run_case(n, route, nyquist)
    assert np.max(np.abs(res.u - ref)) < TOL * np.linalg.norm(u0)


@pytest.mark.slow
@pytest.mark.parametrize("nyquist", ["zero", "negative"])
def test_route_b_explicit_ancillas_n5(nyquist: str) -> None:
    u0, res, ref = run_case(5, "B", nyquist)  # 5 + 15 qubits
    assert np.max(np.abs(res.u - ref)) < TOL * np.linalg.norm(u0)


# ------------------------------------------------------------------ negative controls


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_wrong_decay_rate_is_detected(route: str) -> None:
    u0, res, _ = run_case(3, route, "zero")
    wrong = classical.evolve_spectral(u0, T, C, 1.05 * NU, L, "zero")
    assert np.max(np.abs(res.u - wrong)) > 1e-3


@pytest.mark.parametrize("route", ["A", "B", "C"])
def test_wrong_advection_sign_is_detected(route: str) -> None:
    u0, res, _ = run_case(3, route, "zero")
    wrong = classical.evolve_spectral(u0, T, -C, NU, L, "zero")
    assert np.max(np.abs(res.u - wrong)) > 1e-2


def test_wrong_nyquist_convention_is_detected() -> None:
    n = 4
    u0 = field(n)
    res = oneshot_sim.evolve_oneshot(u0, C, NU, L, T, "A", "negative")
    other = classical.evolve_spectral(u0, T, C, NU, L, "zero")
    assert np.max(np.abs(res.u - other)) > 1e-3


def test_route_c_with_too_narrow_a_window_fails() -> None:
    """The window is a real truncation: a loose tail tolerance must show up as error."""
    n = 5
    u0 = field(n)
    loose = oneshot_sim.evolve_oneshot(u0, C, 0.2, L, T, "C", "zero", tail_tol=0.3)
    ref = classical.evolve_spectral(u0, T, C, 0.2, L, "zero")
    assert np.max(np.abs(loose.u - ref)) > 1e-3


# ------------------------------------------------------------------ route C scaling


def test_route_c_width_register_cost_is_exponential_in_n_at_fixed_time() -> None:
    a = oneshot.diffusion_exponent(NU, T, L)
    q = [oneshot.gaussian_plan(n, a).q for n in range(3, 10)]
    cost = [oneshot.gaussian_plan(n, a).prep_r.n_ry for n in range(3, 10)]
    assert all(q2 >= q1 for q1, q2 in zip(q, q[1:], strict=False))
    assert q[-1] > q[0]  # the width register grows with n
    assert cost[-1] / cost[0] >= 8  # preparation cost ~ 2**q ~ sigma
    # sigma doubles per qubit, so for the largest n the window reaches the full grid
    assert q[-1] == 9 or oneshot.gaussian_plan(9, a).dropped <= 1e-13


def test_route_c_kernel_is_not_compact_below_about_two_and_a_half_cells() -> None:
    """``g`` is cut at the Nyquist mode, ``g(N/2) = exp(-pi^2 sigma^2 / 2)``. For small
    ``sigma`` that is not negligible, the real-space kernel has a slowly decaying
    ripple, and a 1e-13 window needs the whole grid (``q = n``). For ``sigma >= 4``
    the window is compact (``q ~ log2 sigma + 4``)."""
    n = 8
    N = 2**n
    for sigma, compact in [(1.0, False), (2.0, False), (4.0, True), (8.0, True)]:
        a = 2 * np.pi**2 * sigma**2 / N**2
        plan = oneshot.gaussian_plan(n, a)
        assert (plan.q < n) == compact, (sigma, plan.q)
        if compact:
            assert (
                plan.q == int(np.ceil(np.log2(sigma))) + 4
                or plan.q == int(np.log2(sigma)) + 4
            )


def test_route_c_window_error_bound_holds() -> None:
    n, a = 6, oneshot.diffusion_exponent(0.02, 1.0, L)
    plan = oneshot.gaussian_plan(n, a, tail_tol=1e-8)
    assert plan.dropped <= 1e-8 and plan.q < n
    assert plan.lam == pytest.approx(1.0, abs=2e-8)  # all weights positive: sum = g(0)


# ----------------------------------------------------- route B1 (mid-circuit)


def b1_args(n: int, nyquist: str = "zero"):
    from cudaq_pde.solvers import oneshot_kernels as k

    u0 = field(n, seed=7)
    psi, _ = encoding.encode(u0)
    plan = oneshot.oneshot_plan(n, C, NU, L, T, nyquist)
    fac = oneshot.contractions(n, plan.a)
    args = (psi.tolist(), plan.angles, plan.phi, plan.zero, fac.c1, fac.c2, fac.theta)
    return u0, plan, fac, k.oneshot_b1_run, args


def test_run_records_is_seeded() -> None:
    _, _, fac, kernel, args = b1_args(3)
    r1 = sampling.run_records(kernel, *args, shots=60, seed=3)
    r2 = sampling.run_records(kernel, *args, shots=60, seed=3)
    assert r1.shape == (60, fac.count + 3)
    assert np.array_equal(r1, r2)


@pytest.mark.slow
def test_b1_mid_circuit_success_probability_and_post_selected_distribution() -> None:
    n, shots = 3, 8000
    u0, plan, fac, kernel, args = b1_args(n)
    rec = sampling.run_records(kernel, *args, shots=shots, seed=11)
    F = fac.count
    ok = rec[:, :F].sum(axis=1) == 0
    p_exp = oneshot.success_probability(u0, plan.a)
    sigma = np.sqrt(p_exp * (1 - p_exp) / shots)
    assert abs(ok.mean() - p_exp) < 4 * sigma
    # same probability as the explicit-ancilla route B2 on the state vector
    b2 = oneshot_sim.evolve_oneshot(u0, C, NU, L, T, "B", "zero")
    assert abs(ok.mean() - b2.p_succ) < 4 * sigma
    # post-selected system distribution is |u(T)|^2 / ||u(T)||^2
    bits = rec[ok][:, F:]
    idx = (bits * (1 << np.arange(n))).sum(axis=1)
    counts = np.bincount(idx, minlength=2**n)
    ref = classical.evolve_spectral(u0, T, C, NU, L, "zero")
    p = ref**2 / np.sum(ref**2)
    stat = np.sum((counts - counts.sum() * p) ** 2 / (counts.sum() * p))
    assert stat < chi2.ppf(1 - 1e-4, 2**n - 1)
