"""Test-only wrappers: prepare a basis state, apply one shift variant."""

import cudaq

from cudaq_pde.solvers.shift_kernels import (
    decrement_ladder,
    decrement_mcx,
    increment_ladder,
    increment_mcx,
    shift_qft,
)


@cudaq.kernel
def apply_shift_variant(
    n: int, m: int, bits: list[int], kind: int, angles: list[float]
):
    """kind: 0/1 inc/dec qft (with matching angles), 2/3 mcx, 4/5 ladder; m work."""
    q = cudaq.qvector(n + m)
    sys = q.front(n)
    work = q.back(m)
    for b in range(n):
        if bits[b] == 1:
            x(sys[b])
    if kind == 0 or kind == 1:
        shift_qft(sys, angles)
    if kind == 2:
        increment_mcx(sys)
    if kind == 3:
        decrement_mcx(sys)
    if kind == 4:
        increment_ladder(sys, work)
    if kind == 5:
        decrement_ladder(sys, work)
