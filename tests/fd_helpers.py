"""Test helper: dense central-difference operator built from the stencil."""

import numpy as np


def dense_L(n: int, c: float, nu: float, L: float) -> np.ndarray:
    """Dense ``Lfd`` built from the stencil, not from the shift coefficients."""
    N = 2**n
    h = L / N
    eye = np.eye(N)
    up = np.roll(eye, 1, axis=1)  # (up @ u)_j = u_{j+1}
    down = np.roll(eye, -1, axis=1)  # (down @ u)_j = u_{j-1}
    probe = np.arange(N, dtype=float)
    np.testing.assert_array_equal(up @ probe, np.roll(probe, -1))
    d1 = (up - down) / (2 * h)
    d2 = (up - 2 * eye + down) / h**2
    return -c * d1 + nu * d2
