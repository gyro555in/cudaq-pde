"""Test-only wrappers: prepare a basis state, apply one shift variant."""

import cudaq

from cudaq_pde.solvers.shift_kernels import (
    decrement_ladder,
    decrement_mcx,
    decrement_qft,
    increment_ladder,
    increment_mcx,
    increment_qft,
)


@cudaq.kernel
def apply_shift_variant(n: int, m: int, bits: list[int], kind: int):
    """kind: 0/1 inc/dec qft, 2/3 inc/dec mcx, 4/5 inc/dec ladder; m work qubits."""
    q = cudaq.qvector(n + m)
    sys = q.front(n)
    work = q.back(m)
    for b in range(n):
        if bits[b] == 1:
            x(sys[b])
    if kind == 0:
        increment_qft(sys)
    if kind == 1:
        decrement_qft(sys)
    if kind == 2:
        increment_mcx(sys)
    if kind == 3:
        decrement_mcx(sys)
    if kind == 4:
        increment_ladder(sys, work)
    if kind == 5:
        decrement_ladder(sys, work)
