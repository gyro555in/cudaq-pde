"""The sampling wrapper: counts, seeding, ledger (qpp-cpu)."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde.solvers.sampling import Sampler, sample_counts
from ordering_kernels import basis_state
from spectral_test_kernels import apply_qft_piece


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def test_counts_sum_to_shots_and_bitstring_order() -> None:
    counts = sample_counts(basis_state, 3, 1, shots=500, seed=1)
    assert counts == {"010": 500}  # character k is qubit k


def test_same_seed_reproduces_and_other_seed_differs() -> None:
    # a state with a broad output distribution, so the counts really are random
    v = np.arange(1.0, 9.0)
    amps = (v / np.linalg.norm(v)).astype(complex).tolist()
    a = sample_counts(apply_qft_piece, amps, 0, shots=200, seed=5)
    b = sample_counts(apply_qft_piece, amps, 0, shots=200, seed=5)
    c = sample_counts(apply_qft_piece, amps, 0, shots=200, seed=6)
    assert a == b  # same seed, same counts
    assert a != c  # another seed, another realization
    assert sum(a.values()) == sum(c.values()) == 200


def test_sampler_ledger_and_seed_progression() -> None:
    s = Sampler(base_seed=10)
    s.counts(basis_state, 3, 0, shots=100)
    s.counts(basis_state, 3, 2, shots=250)
    assert (s.circuits, s.shots) == (2, 350)


def test_rejects_zero_shots() -> None:
    with pytest.raises(ValueError):
        sample_counts(basis_state, 3, 0, shots=0, seed=0)
