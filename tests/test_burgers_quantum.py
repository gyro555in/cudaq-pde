"""CP8 quantum heat step: statevector agreement (STATEVECTOR CHECK, test only) and the
sampled readout against the classical Cole-Hopf distribution."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest
from scipy.stats import chi2

from burgers_test_kernels import STATE_KERNELS
from cudaq_pde import burgers as B
from cudaq_pde import classical
from cudaq_pde.solvers import burgers_quantum as bq
from cudaq_pde.solvers import oneshot
from cudaq_pde.solvers.spectral_measure import (
    bit_reverse_index,
    classical_power_spectrum,
)

L = 2 * np.pi
ROUTES = ["A", "B2", "C"]


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def case(n: int, nu: float, T: float = 1.0, mean: float = 0.0):
    u0 = mean + np.sin(classical.grid(n)) + 0.5 * np.sin(2 * classical.grid(n))
    return B.discrete_burgers(u0, nu, T)


def accepted_block(route: str, d: B.DiscreteBurgers, nu, T, fourier=False):
    call = bq.heat_call(d.phi0, nu, T, L, route, fourier)
    state = np.array(cudaq.get_state(STATE_KERNELS[route], *call.args))
    return state[: 2**call.n], call


# ------------------------------------------------------------------ statevector


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("n,nu", [(3, 0.5), (4, 0.5), (4, 0.1)])
def test_heat_step_equals_the_classical_heat_solution(route, n, nu) -> None:
    T = 0.8
    d = case(n, nu, T)
    block, call = accepted_block(route, d, nu, T)
    norm0 = np.linalg.norm(d.phi0)
    got = call.lam * norm0 * block
    assert np.max(np.abs(got.imag)) < 1e-12 * norm0
    assert np.max(np.abs(got.real - d.phiT)) < 1e-12 * norm0
    p_succ = np.vdot(block, block).real
    assert p_succ == pytest.approx(d.p_succ / call.lam**2, abs=1e-12)


@pytest.mark.parametrize("route", ROUTES)
def test_fourier_variant_reads_the_bit_reversed_spectrum(route: str) -> None:
    n, nu, T = 4, 0.5, 0.8
    d = case(n, nu, T)
    block, call = accepted_block(route, d, nu, T, fourier=True)
    prob = np.abs(block) ** 2  # by register index
    fft_order = np.zeros(2**n)
    for r in range(2**n):
        fft_order[bit_reverse_index(r, n)] = prob[r]
    expect = classical_power_spectrum(d.phiT) * d.p_succ / call.lam**2
    assert np.max(np.abs(fft_order - expect)) < 1e-12


def test_zero_time_is_the_identity() -> None:
    d = case(3, 0.5, 0.0)
    block, _ = accepted_block("A", d, 0.5, 0.0)
    assert np.vdot(block, block).real == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(np.abs(block) ** 2, d.p, atol=1e-14)


def test_wrong_viscosity_is_detected() -> None:
    d = case(4, 0.5, 0.8)
    block, _ = accepted_block("A", d, 0.5 * 1.2, 0.8)  # circuit built with nu * 1.2
    got = np.linalg.norm(d.phi0) * block.real
    assert np.max(np.abs(got - d.phiT)) > 1e-3


# ------------------------------------------------------------------ sampled


def gof(counts: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Chi-square statistic and its 1e-4 critical value (tiny bins dropped)."""
    keep = p > 1e-12
    M = counts.sum()
    stat = float(np.sum((counts[keep] - M * p[keep]) ** 2 / (M * p[keep])))
    return stat, float(chi2.ppf(1 - 1e-4, keep.sum() - 1))


@pytest.mark.parametrize("route", ROUTES)
def test_sampled_acceptance_and_distribution(route: str) -> None:
    n, nu, T, shots = 3, 0.5, 0.8, 20_000
    d = case(n, nu, T)
    s = bq.sample_heat(d.phi0, nu, T, L, route, shots, seed=5)
    p_exp = bq.expected_p_succ(d.phi0, nu, T, L, route)
    assert p_exp == pytest.approx(d.p_succ / s.lam**2, abs=1e-12)
    sigma = np.sqrt(p_exp * (1 - p_exp) / shots)
    assert abs(s.p_succ_hat - p_exp) < 4 * sigma
    stat, crit = gof(s.grid_counts, d.p)
    assert stat < crit


def test_sampling_is_seeded() -> None:
    d = case(3, 0.5)
    a = bq.sample_heat(d.phi0, 0.5, 1.0, L, "A", 500, seed=3)
    b = bq.sample_heat(d.phi0, 0.5, 1.0, L, "A", 500, seed=3)
    assert np.array_equal(a.bin_counts, b.bin_counts)


def test_sampled_distribution_rejects_a_wrong_viscosity() -> None:
    n, nu, T = 3, 0.5, 0.8
    d = case(n, nu, T)
    wrong = case(n, 1.2 * nu, T)
    s = bq.sample_heat(d.phi0, nu, T, L, "A", 20_000, seed=5)
    stat, crit = gof(s.grid_counts, wrong.p)
    assert stat > 3 * crit


@pytest.mark.parametrize("route", ROUTES)
def test_sampled_spectrum_matches_the_classical_one(route: str) -> None:
    n, nu, T, shots = 3, 0.5, 0.8, 20_000
    d = case(n, nu, T)
    s = bq.sample_heat(d.phi0, nu, T, L, route, shots, seed=9, fourier=True)
    ref = classical_power_spectrum(d.phiT)
    stat, crit = gof(s.fft_counts(), ref)
    assert stat < crit


def test_readout_modes_are_not_interchangeable() -> None:
    d = case(3, 0.5)
    pos = bq.sample_heat(d.phi0, 0.5, 1.0, L, "A", 200, seed=1)
    four = bq.sample_heat(d.phi0, 0.5, 1.0, L, "A", 200, seed=1, fourier=True)
    with pytest.raises(ValueError):
        pos.fft_counts()
    with pytest.raises(ValueError):
        four.grid_counts  # noqa: B018


def test_routes_reject_unknown_names_and_use_expected_ancillas() -> None:
    d = case(4, 0.5)
    with pytest.raises(ValueError):
        bq.heat_call(d.phi0, 0.5, 1.0, L, "B1")
    assert bq.heat_call(d.phi0, 0.5, 1.0, L, "A").n_anc == 1
    assert bq.heat_call(d.phi0, 0.5, 1.0, L, "B2").n_anc == 10  # n(n+1)/2
    a = oneshot.diffusion_exponent(0.5, 1.0, L)
    assert bq.heat_call(d.phi0, 0.5, 1.0, L, "C").n_anc == oneshot.gaussian_plan(4, a).q
