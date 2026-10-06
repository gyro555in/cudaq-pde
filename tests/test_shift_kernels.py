"""Shift kernels vs numpy roll, all basis states, n = 2..6, qpp-cpu fp64.

STATEVECTOR CHECKS (test only): the kernels themselves never read a state.
"""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from shift_test_kernels import apply_shift_variant

TOL = 1e-12
NS = range(2, 7)
KINDS = {  # kind -> (variant, direction)
    0: ("qft", +1),
    1: ("qft", -1),
    2: ("mcx", +1),
    3: ("mcx", -1),
    4: ("ladder", +1),
    5: ("ladder", -1),
}


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def work_qubits(n: int, variant: str) -> int:
    """Ancillas allocated by the wrapper (the ladder needs n - 2, at least 1 spare)."""
    return max(n - 2, 1)


def shift_matrix(n: int, kind: int) -> np.ndarray:
    """Matrix of the kernel on the system register (ancillas must return to |0>)."""
    N, m = 2**n, work_qubits(n, KINDS[kind][0])
    cols = []
    for j in range(N):
        bits = [(j >> b) & 1 for b in range(n)]
        state = np.array(cudaq.get_state(apply_shift_variant, n, m, bits, kind))
        sys_part, rest = state[:N], state[N:]
        assert np.max(np.abs(rest)) < TOL, "ancilla not returned to |0>"
        cols.append(sys_part)
    return np.array(cols).T


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("n", NS)
def test_every_basis_state_matches_numpy_roll(n: int, kind: int) -> None:
    variant, direction = KINDS[kind]
    M = shift_matrix(n, kind)
    N = 2**n
    for j in range(N):
        expected = np.roll(np.eye(N)[j], direction)  # e_{(j + direction) mod N}
        np.testing.assert_allclose(M[:, j], expected, atol=TOL)


@pytest.mark.parametrize("variant", ["qft", "mcx", "ladder"])
@pytest.mark.parametrize("n", NS)
def test_decrement_is_inverse_of_increment(n: int, variant: str) -> None:
    kinds = {v: k for k, (v, d) in KINDS.items() if d == +1}
    inc, dec = shift_matrix(n, kinds[variant]), shift_matrix(n, kinds[variant] + 1)
    np.testing.assert_allclose(dec @ inc, np.eye(2**n), atol=TOL)
    np.testing.assert_allclose(dec, inc.T, atol=TOL)


@pytest.mark.parametrize("n", [3, 5])
def test_naming_matches_the_fd_convention_of_cp4(n: int) -> None:
    """CP4's S+ (u_{j+1}) is decrement and S- (u_{j-1}) is increment."""
    u = np.random.default_rng(n).standard_normal(2**n)
    np.testing.assert_allclose(shift_matrix(n, 3) @ u, np.roll(u, -1), atol=TOL)
    np.testing.assert_allclose(shift_matrix(n, 2) @ u, np.roll(u, +1), atol=TOL)
