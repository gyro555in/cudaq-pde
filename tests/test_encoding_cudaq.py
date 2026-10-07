from collections.abc import Iterator
from functools import reduce

import cudaq
import numpy as np
import pytest

from cudaq_pde import encoding
from cudaq_pde.encoding_kernels import prepare_amplitudes
from ordering_kernels import basis_state, product_ry

NS = [2, 3, 4, 5, 6]


@pytest.fixture
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def _angles(n: int) -> list[float]:
    # Distinct, non-palindromic so a reversed qubit order cannot match.
    return [0.3 + 0.7 * k + 0.05 * k**2 for k in range(n)]


def _numpy_product_state(angles: list[float]) -> np.ndarray:
    factors = [np.array([np.cos(t / 2), np.sin(t / 2)]) for t in angles]
    # np.kron puts its first argument in the most significant position.
    return reduce(np.kron, factors[::-1])


@pytest.mark.usefixtures("cpu_target")
@pytest.mark.parametrize("n", NS)
def test_basis_state_index_is_power_of_two(n: int) -> None:
    for k in range(n):
        sv = np.array(cudaq.get_state(basis_state, n, k))
        expected = np.zeros(2**n, dtype=complex)
        expected[2**k] = 1.0
        np.testing.assert_allclose(sv, expected, atol=1e-14)


@pytest.mark.usefixtures("cpu_target")
@pytest.mark.parametrize("n", NS)
def test_product_state_matches_numpy_kron(n: int) -> None:
    angles = _angles(n)
    sv = np.array(cudaq.get_state(product_ry, angles))
    assert sv.dtype == np.complex128
    np.testing.assert_allclose(sv, _numpy_product_state(angles), atol=1e-14)
    reversed_ref = _numpy_product_state(angles[::-1])
    assert np.max(np.abs(sv - reversed_ref)) > 1e-2


@pytest.mark.usefixtures("cpu_target")
@pytest.mark.parametrize("n", NS)
def test_amplitude_round_trip_through_cudaq(n: int) -> None:
    rng = np.random.default_rng(200 + n)
    u = rng.standard_normal(2**n) * 2.5
    psi, norm = encoding.encode(u)
    sv = np.array(cudaq.get_state(prepare_amplitudes, psi.tolist()))
    np.testing.assert_allclose(sv, psi, atol=1e-14)
    np.testing.assert_allclose(encoding.decode(sv, norm), u, atol=1e-14)


@pytest.mark.usefixtures("cpu_target")
@pytest.mark.parametrize("n", NS)
def test_sample_bitstring_maps_to_index(n: int) -> None:
    for k in range(n):
        counts = cudaq.sample(basis_state, n, k, shots_count=20)
        assert len(counts) == 1
        s = next(iter(counts))
        assert s[k] == "1"
        assert s.count("1") == 1
        assert encoding.bitstring_to_index(s) == 2**k


@pytest.mark.gpu
def test_ordering_on_gpu_fp64() -> None:
    cudaq.set_target("nvidia", option="fp64")
    try:
        n = 4
        angles = _angles(n)
        sv = np.array(cudaq.get_state(product_ry, angles))
        np.testing.assert_allclose(sv, _numpy_product_state(angles), atol=1e-12)
        for k in range(n):
            s = next(iter(cudaq.sample(basis_state, n, k, shots_count=10)))
            assert encoding.bitstring_to_index(s) == 2**k
    finally:
        cudaq.reset_target()
