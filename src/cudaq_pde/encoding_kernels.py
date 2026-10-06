"""Kernels for loading classical data into a qubit register.

Qubit ordering follows :mod:`cudaq_pde.encoding`: qubit 0 is the least
significant bit of the grid index.
"""

import cudaq


@cudaq.kernel
def prepare_amplitudes(amps: list[complex]):
    """Prepare the normalized state ``amps`` on ``log2(len(amps))`` qubits.

    Resource caveat: on a QPU the compiler lowers this to a generic state
    preparation circuit with O(2**n) gates, so it is a simulator convenience
    and a cost to account for, not a scalable loader.
    """
    cudaq.qvector(amps)
