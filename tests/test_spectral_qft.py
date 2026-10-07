"""Pin the QFT sign and bit order against numpy (n = 2..6, all basis inputs)."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from spectral_test_kernels import apply_qft_piece

NS = [2, 3, 4, 5, 6]
TOL = 1e-12


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def _matrices(n: int) -> dict[str, np.ndarray]:
    N = 2**n
    f_plus = np.sqrt(N) * np.fft.ifft(np.eye(N), axis=0)  # exp(+2 pi i jk/N)/sqrt(N)
    f_minus = np.fft.fft(np.eye(N), axis=0) / np.sqrt(N)
    rev = np.zeros((N, N))
    for j in range(N):
        rev[int(format(j, f"0{n}b")[::-1], 2), j] = 1.0
    return {"F+": f_plus, "F-": f_minus, "R": rev}


def _circuit_matrix(n: int, mode: int) -> np.ndarray:
    N = 2**n
    cols = [
        np.array(cudaq.get_state(apply_qft_piece, np.eye(N)[j].astype(complex), mode))
        for j in range(N)
    ]
    return np.array(cols).T


def test_numpy_identities() -> None:
    m = _matrices(3)
    np.testing.assert_allclose(m["F+"] @ m["F-"], np.eye(8), atol=TOL)
    np.testing.assert_allclose(m["F-"], m["F+"].conj(), atol=TOL)


@pytest.mark.parametrize("n", NS)
def test_b_is_fplus_times_reversal(n: int) -> None:
    m = _matrices(n)
    np.testing.assert_allclose(_circuit_matrix(n, 0), m["F+"] @ m["R"], atol=TOL)


@pytest.mark.parametrize("n", NS)
def test_b_dagger_is_reversal_times_fminus(n: int) -> None:
    m = _matrices(n)
    np.testing.assert_allclose(_circuit_matrix(n, 1), m["R"] @ m["F-"], atol=TOL)


@pytest.mark.parametrize("n", NS)
def test_qft_with_swaps_is_sqrtN_ifft(n: int) -> None:
    m = _matrices(n)
    np.testing.assert_allclose(_circuit_matrix(n, 2), m["F+"], atol=TOL)


@pytest.mark.parametrize("n", NS)
def test_qft_dagger_with_swaps_is_fft_over_sqrtN(n: int) -> None:
    m = _matrices(n)
    np.testing.assert_allclose(_circuit_matrix(n, 3), m["F-"], atol=TOL)
