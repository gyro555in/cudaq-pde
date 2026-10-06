import numpy as np
import pytest

from cudaq_pde import encoding


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_round_trip(n: int) -> None:
    rng = np.random.default_rng(100 + n)
    u = rng.standard_normal(2**n) * 3.7
    psi, norm = encoding.encode(u)
    assert psi.dtype == np.complex128
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-15
    assert norm == pytest.approx(np.linalg.norm(u), rel=1e-15)
    np.testing.assert_allclose(encoding.decode(psi, norm), u, atol=1e-14)


def test_amplitude_order_is_grid_order() -> None:
    u = np.array([3.0, 0.0, 4.0, 0.0])
    psi, norm = encoding.encode(u)
    np.testing.assert_allclose(psi, [0.6, 0, 0.8, 0])
    assert norm == 5.0


def test_num_qubits() -> None:
    assert [encoding.num_qubits(2**n) for n in range(1, 7)] == [1, 2, 3, 4, 5, 6]
    for bad in (0, 1, 3, 12):
        with pytest.raises(ValueError):
            encoding.num_qubits(bad)


def test_encode_validation() -> None:
    with pytest.raises(ValueError):
        encoding.encode(np.zeros(4))
    with pytest.raises(ValueError):
        encoding.encode(np.ones(6))
    with pytest.raises(ValueError):
        encoding.encode(np.array([1.0, np.inf, 0.0, 0.0]))
    with pytest.raises(ValueError):
        encoding.encode(np.ones(4, dtype=complex))
    with pytest.raises(ValueError):
        encoding.encode(np.ones((2, 2)))


def test_decode_rejects_complex_state() -> None:
    psi = np.array([1, 1j, 0, 0]) / np.sqrt(2)
    with pytest.raises(ValueError):
        encoding.decode(psi, 1.0)


@pytest.mark.parametrize("n", [2, 3, 6])
def test_index_bits_inverse(n: int) -> None:
    for j in range(2**n):
        assert encoding.bits_to_index(encoding.index_to_bits(j, n)) == j


def test_bit_order_is_lsb_first() -> None:
    assert encoding.index_to_bits(1, 3) == (1, 0, 0)
    assert encoding.index_to_bits(4, 3) == (0, 0, 1)
    assert encoding.bitstring_to_index("100") == 1
    assert encoding.bitstring_to_index("001") == 4
    assert encoding.bitstring_to_index("110") == 3


def test_index_bits_validation() -> None:
    with pytest.raises(ValueError):
        encoding.index_to_bits(8, 3)
    with pytest.raises(ValueError):
        encoding.bits_to_index([0, 2])
    with pytest.raises(ValueError):
        encoding.bitstring_to_index("01x")
    with pytest.raises(ValueError):
        encoding.bitstring_to_index("")
