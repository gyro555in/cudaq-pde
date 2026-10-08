"""Shot-based sampling: the single place that calls ``cudaq.sample``.

CUDA-Q 0.16 prints a FutureWarning that the ``sample`` and ``observe`` primitives
"will change in a future release". Every sampled quantity in the hardware path goes
through :func:`sample_counts`, so a future API change touches this one function.

Seeding: :func:`sample_counts` seeds the simulator RNG with the ``seed`` it is given.
:class:`Sampler` hands out ``base_seed + circuit_index``, so a run is reproducible
from ``base_seed`` alone while no two circuits reuse one noise realization.
"""

from __future__ import annotations

from dataclasses import dataclass

import cudaq
import numpy as np


def sample_counts(kernel, *args, shots: int, seed: int) -> dict[str, int]:
    """Run ``kernel`` for ``shots`` shots and return ``{bitstring: count}``.

    Bitstring character ``k`` is qubit ``k`` (see :mod:`cudaq_pde.encoding`).
    """
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    cudaq.set_random_seed(seed)
    result = cudaq.sample(kernel, *args, shots_count=shots)
    return {bits: int(count) for bits, count in result.items()}


@dataclass
class Sampler:
    """Seeded sampling with a ledger of circuits and shots spent."""

    base_seed: int = 0
    circuits: int = 0
    shots: int = 0

    def counts(self, kernel, *args, shots: int) -> dict[str, int]:
        out = sample_counts(
            kernel, *args, shots=shots, seed=self.base_seed + self.circuits
        )
        self.circuits += 1
        self.shots += shots
        return out


def run_records(kernel, *args, shots: int, seed: int) -> np.ndarray:
    """Run a ``-> list[bool]`` kernel with ``cudaq.run``; one row of bits per shot.

    Mid-circuit measurements must be returned by the kernel to be seen per shot:
    ``cudaq.sample`` keeps only the final register, so joint statistics of
    mid-circuit outcomes (route B1) need ``cudaq.run``. Seeding as in
    :func:`sample_counts`.
    """
    if shots < 1:
        raise ValueError(f"shots must be >= 1, got {shots}")
    cudaq.set_random_seed(seed)
    out = cudaq.run(kernel, *args, shots_count=shots)
    return np.array([[int(b) for b in row] for row in out], dtype=np.int8)
