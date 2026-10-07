"""Measurable power spectrum: ordering, evolution, and shot-noise scaling (qpp-cpu).

STATEVECTOR CHECKS (test only) are marked; the estimator itself only samples.
"""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde import classical
from cudaq_pde.solvers import spectral_measure as sm
from cudaq_pde.solvers.sampling import Sampler
from cudaq_pde.solvers.spectral import phase_angles
from cudaq_pde.solvers.spectral_kernels import (
    spectrum_circuit,
    spectrum_circuit_fused,
    spectrum_circuit_gates,
)

L = classical.TWO_PI


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def pulse(n: int) -> np.ndarray:
    x = classical.grid(n, L)
    d = (x - np.pi + np.pi) % L - np.pi
    return np.exp(-0.5 * (d / 0.5) ** 2) + 0.2 * np.cos(3 * x)


def evolved_reference(u0, c, T, nyquist):
    return classical.evolve_spectral(u0, T, c, 0.0, L, nyquist=nyquist)


# --------------------------------------------------------- exact distribution


def exact_probabilities(circuit, args, n: int) -> np.ndarray:
    """STATEVECTOR CHECK (test only): outcome probabilities in FFT index order."""
    state = np.array(cudaq.get_state(circuit, *args))
    p_reg = np.abs(state) ** 2
    p = np.zeros(2**n)
    for r in range(2**n):
        p[sm.bit_reverse_index(r, n)] = p_reg[r]
    return p


@pytest.mark.parametrize("nyquist", ["zero", "negative"])
@pytest.mark.parametrize("swaps", [False, True])
@pytest.mark.parametrize("n", range(3, 9))
def test_circuit_distribution_is_the_classical_power_spectrum(
    n, swaps, nyquist
) -> None:
    u0, c, T = pulse(n), 1.3, 2.9
    angles, phi = phase_angles(n, c, L, T)
    psi = (u0 / np.linalg.norm(u0)).astype(complex).tolist()
    p = exact_probabilities(
        spectrum_circuit, (psi, angles, phi, nyquist == "zero", swaps, 1), n
    )
    ref = sm.classical_power_spectrum(evolved_reference(u0, c, T, nyquist))
    np.testing.assert_allclose(p, ref, atol=1e-12)
    assert p.sum() == pytest.approx(1.0, abs=1e-12)
    # advection leaves the spectrum invariant: the same as the initial field's
    np.testing.assert_allclose(ref, sm.classical_power_spectrum(u0), atol=1e-12)


@pytest.mark.parametrize("n", [3, 5, 7])
def test_fused_circuit_has_the_same_distribution_and_fewer_gates(n: int) -> None:
    u0, c, T = pulse(n), 0.8, 1.7
    angles, phi = phase_angles(n, c, L, T)
    psi = (u0 / np.linalg.norm(u0)).astype(complex).tolist()
    full = exact_probabilities(spectrum_circuit, (psi, angles, phi, True, False, 1), n)
    fused = exact_probabilities(spectrum_circuit_fused, (psi, angles, phi, True, 1), n)
    np.testing.assert_allclose(fused, full, atol=1e-12)
    a = cudaq.estimate_resources(
        spectrum_circuit_gates, n, angles, phi, True, False, 1, False
    )
    b = cudaq.estimate_resources(
        spectrum_circuit_gates, n, angles, phi, True, False, 1, True
    )
    # the QFT pair cancels: 2 QFTs (n h + n(n-1)/2 cr1 each) fewer
    saved = sum(dict(a.to_dict()).values()) - sum(dict(b.to_dict()).values())
    assert saved == 2 * (n + n * (n - 1) // 2)


# ------------------------------------------------------------------- ordering


def test_single_mode_peaks_land_on_the_right_fft_bins() -> None:
    """u = 1 + cos(3x): peaks at FFT bins 0, 3 and N - 3 (NOT the bit-reversed ones)."""
    n, shots = 4, 200_000
    x = classical.grid(n, L)
    u0 = 1.0 + np.cos(3 * x)
    p = sm.estimate_power_spectrum(u0, 1.0, L, 0.7, shots, seed=1)
    ref = sm.classical_power_spectrum(u0)
    assert set(np.flatnonzero(ref > 1e-12)) == {0, 3, 13}
    assert set(np.flatnonzero(p > 1e-4)) == {0, 3, 13}
    np.testing.assert_allclose(p, ref, atol=5 * np.sqrt(0.25 / shots))
    # negative control: reading the register index directly would be wrong
    wrong = np.zeros(2**n)
    for k in (0, 3, 13):
        wrong[sm.bit_reverse_index(k, n)] = ref[k]
    assert np.max(np.abs(wrong - ref)) > 0.1


def test_nyquist_content_is_measured_at_the_nyquist_bin() -> None:
    n, shots = 4, 100_000
    u0 = np.cos(np.pi * np.arange(2**n)) + 2.0  # DC plus the Nyquist mode
    p = sm.estimate_power_spectrum(u0, 1.0, L, 0.9, shots, seed=2, nyquist="negative")
    ref = sm.classical_power_spectrum(u0)
    assert ref[2 ** (n - 1)] > 0.1
    np.testing.assert_allclose(p, ref, atol=5 * np.sqrt(0.25 / shots))


def test_bit_reverse_and_counts_reduction() -> None:
    assert [sm.bit_reverse_index(r, 3) for r in range(8)] == [0, 4, 2, 6, 1, 5, 3, 7]
    # string "100" is qubit 0 set: register index 1, FFT index rev(1) = 4
    p = sm.counts_to_power_spectrum({"100": 30, "000": 70}, 3, 100)
    assert p[4] == pytest.approx(0.3) and p[0] == pytest.approx(0.7)


# --------------------------------------------------------------- shot scaling


def rms_errors(n: int, shots: int, seeds: int, base: int = 0) -> np.ndarray:
    u0, c, T = pulse(n), 1.0, 1.0
    ref = sm.classical_power_spectrum(evolved_reference(u0, c, T, "zero"))
    errs = []
    for s in range(seeds):
        p = sm.estimate_power_spectrum(u0, c, L, T, shots, seed=base + s)
        errs.append(np.sqrt(np.sum((p - ref) ** 2)))
    return np.array(errs), ref


@pytest.mark.parametrize("n", [3, 5, 7, 8])
def test_error_matches_the_multinomial_prediction(n: int) -> None:
    shots, seeds = 20_000, 40
    errs, ref = rms_errors(n, shots, seeds, base=100 * n)
    rms = np.sqrt(np.mean(errs**2))
    assert rms == pytest.approx(sm.expected_rms_error(ref, shots), rel=0.12)


@pytest.mark.parametrize("n", [3, 4, 6])
def test_error_scales_as_inverse_sqrt_shots(n: int) -> None:
    shots_list = [1_000, 10_000, 100_000]
    rms = [
        np.sqrt(np.mean(rms_errors(n, s, 25, base=7 * n)[0] ** 2)) for s in shots_list
    ]
    slope = np.polyfit(np.log(shots_list), np.log(rms), 1)[0]
    assert -0.58 < slope < -0.42, (slope, rms)


def test_the_estimator_never_reads_a_state(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("cudaq.get_state called by the spectrum estimator")

    monkeypatch.setattr(cudaq, "get_state", forbidden)
    p = sm.estimate_power_spectrum(pulse(4), 1.0, L, 0.5, 500, seed=0)
    assert p.sum() == pytest.approx(1.0)
    assert Sampler().circuits == 0
