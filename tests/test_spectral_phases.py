"""Phase-layer angles vs the classical wavenumbers (no cudaq needed)."""

import numpy as np
import pytest

from cudaq_pde import classical
from cudaq_pde.solvers.spectral import phase_angles

L_VALUES = [classical.TWO_PI, 3.7]


@pytest.mark.parametrize("L", L_VALUES)
@pytest.mark.parametrize("n", range(2, 9))
def test_angles_reproduce_wavenumbers(n: int, L: float) -> None:
    c, dt = -0.7, 1.3
    angles, phi = phase_angles(n, c, L, dt)
    assert len(angles) == n
    N = 2**n
    for nyquist in ("negative", "zero"):
        k = classical.advection_wavenumbers(n, L, nyquist)
        for j in range(N):
            theta = sum(angles[b] for b in range(n) if (j >> b) & 1)
            if nyquist == "zero" and j == N // 2:
                theta += phi
            np.testing.assert_allclose(
                np.exp(1j * theta), np.exp(-1j * c * k[j] * dt), atol=1e-12
            )


def test_top_qubit_has_negative_weight() -> None:
    angles, _ = phase_angles(4, 1.0, classical.TWO_PI, 1.0)
    np.testing.assert_allclose(angles, [-1, -2, -4, 8])


def test_angles_linear_in_dt() -> None:
    a1, p1 = phase_angles(5, 0.9, 3.7, 0.2)
    a3, p3 = phase_angles(5, 0.9, 3.7, 0.6)
    np.testing.assert_allclose(np.array(a3), 3 * np.array(a1), atol=1e-14)
    assert p3 == pytest.approx(3 * p1)


def test_needs_two_qubits() -> None:
    with pytest.raises(ValueError):
        phase_angles(1, 1.0)
