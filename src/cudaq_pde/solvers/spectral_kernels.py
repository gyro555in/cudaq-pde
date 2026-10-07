"""Gate-level kernels for the spectral advection circuit.

Conventions (qubit 0 is the least significant bit of the grid index, see
:mod:`cudaq_pde.encoding`; measured against numpy in the tests):

* ``qft_b`` is the textbook QFT that starts with H on qubit 0 and has no final
  swaps. As a matrix it equals ``F+ R`` with ``F+ = sqrt(N) * ifft`` (sign +,
  the inverse DFT up to normalization) and ``R`` the bit reversal.
* The true QFT is therefore ``qft_b`` after ``reverse_qubits``, and the QFT
  dagger is ``reverse_qubits`` after ``iqft_b``.
"""

import cudaq


@cudaq.kernel
def qft_b(q: cudaq.qview):
    """QFT circuit B (H first on qubit 0), without swaps: equals ``F+ R``."""
    n = q.size()
    for a in range(n):
        h(q[a])
        theta = 3.141592653589793
        for b in range(a + 1, n):
            theta = theta / 2.0
            r1.ctrl(theta, q[b], q[a])


@cudaq.kernel
def iqft_b(q: cudaq.qview):
    """Dagger of :func:`qft_b`: equals ``R F-``, ``F-`` being ``F+`` conjugated."""
    n = q.size()
    for i in range(n):
        a = n - 1 - i
        for j in range(n - 1 - a):
            b = n - 1 - j
            theta = 3.141592653589793
            for _ in range(b - a):
                theta = theta / 2.0
            r1.ctrl(-theta, q[b], q[a])
        h(q[a])


@cudaq.kernel
def reverse_qubits(q: cudaq.qview):
    """Bit reversal ``R`` of the register as floor(n/2) swaps."""
    n = q.size()
    for i in range(n // 2):
        swap(q[i], q[n - 1 - i])


@cudaq.kernel
def phase_layer_natural(q: cudaq.qview, angles: list[float]):
    """``R1(angles[b])`` on the qubit holding bit b of the Fourier index."""
    n = q.size()
    for b in range(n):
        r1(angles[b], q[b])


@cudaq.kernel
def phase_layer_mirrored(q: cudaq.qview, angles: list[float]):
    """Same phases with mirrored qubits, for the swap-free circuit.

    Without the QFT bit reversal the register holds the Fourier index bit-
    reversed, so bit b lives on qubit n-1-b.
    """
    n = q.size()
    for b in range(n):
        r1(angles[b], q[n - 1 - b])


@cudaq.kernel
def controlled_phase_layer_mirrored(
    ctl: cudaq.qubit, q: cudaq.qview, angles: list[float]
):
    """:func:`phase_layer_mirrored` with every phase controlled on ``ctl``.

    Used by the LCU shifts: ``c-(QFT D QFT-dagger) = QFT (c-D) QFT-dagger``, so only
    the phase layer needs the control.
    """
    n = q.size()
    for b in range(n):
        r1.ctrl(angles[b], ctl, q[n - 1 - b])


@cudaq.kernel
def nyquist_phase(q: cudaq.qview, phi: float, mirrored: bool):
    """Phase ``exp(i phi)`` on the single Nyquist basis state.

    The Nyquist state has the top Fourier bit set and all others clear. In the
    natural layout that is qubit n-1 set; mirrored, qubit 0 set. Implemented as
    X on the clear qubits, one (n-1)-controlled R1, X again.
    """
    n = q.size()
    if mirrored:
        for i in range(1, n):
            x(q[i])
        r1.ctrl(phi, q.back(n - 1), q[0])
        for i in range(1, n):
            x(q[i])
    else:
        for i in range(n - 1):
            x(q[i])
        r1.ctrl(phi, q.front(n - 1), q[n - 1])
        for i in range(n - 1):
            x(q[i])


@cudaq.kernel
def spectral_advection_on(
    q: cudaq.qview,
    angles: list[float],
    phi: float,
    zero: bool,
    swaps: bool,
    steps: int,
):
    """``steps`` repetitions of QFT-dagger, phase layer, QFT on register ``q``.

    Swap-free (``swaps=False``): ``B-dagger, D', B``. With swaps: ``B-dagger, R,
    D, R, B``. Both equal ``QFT D QFT-dagger`` exactly.
    """
    for _ in range(steps):
        iqft_b(q)
        if swaps:
            reverse_qubits(q)
            phase_layer_natural(q, angles)
            if zero:
                nyquist_phase(q, phi, False)
            reverse_qubits(q)
        else:
            phase_layer_mirrored(q, angles)
            if zero:
                nyquist_phase(q, phi, True)
        qft_b(q)


@cudaq.kernel
def spectral_advection(
    amps: list[complex],
    angles: list[float],
    phi: float,
    zero: bool,
    swaps: bool,
    steps: int,
):
    """Load ``amps`` and apply :func:`spectral_advection_on`."""
    q = cudaq.qvector(amps)
    spectral_advection_on(q, angles, phi, zero, swaps, steps)


@cudaq.kernel
def spectral_advection_circuit(
    n: int,
    angles: list[float],
    phi: float,
    zero: bool,
    swaps: bool,
    steps: int,
):
    """The evolution circuit alone on ``n`` qubits (no state preparation).

    Used for resource counting: loading data is a separate, exponentially
    expensive step.
    """
    q = cudaq.qvector(n)
    spectral_advection_on(q, angles, phi, zero, swaps, steps)


@cudaq.kernel
def spectrum_circuit(
    amps: list[complex],
    angles: list[float],
    phi: float,
    zero: bool,
    swaps: bool,
    steps: int,
):
    """Load ``amps``, evolve (:func:`spectral_advection_on`), then swap-free QFT-dagger.

    Sampling the register gives the power spectrum: ``iqft_b = R F-``, so the outcome
    ``r = sum_q s[q] 2**q`` of a bitstring ``s`` is the BIT-REVERSED FFT index and
    ``P(r) = |u_hat_{rev(r)}|^2 / (N ||u||^2)``. The evolution ends with a QFT and this
    circuit starts with a QFT-dagger, which cancel; :func:`spectrum_circuit_fused`
    drops both. Loading ``amps`` is the simulator convenience of CP2 (it costs
    ``O(2**n)`` gates on hardware), not a scalable loader.
    """
    q = cudaq.qvector(amps)
    spectral_advection_on(q, angles, phi, zero, swaps, steps)
    iqft_b(q)


@cudaq.kernel
def spectrum_circuit_fused(
    amps: list[complex],
    angles: list[float],
    phi: float,
    zero: bool,
    steps: int,
):
    """Same measurement distribution as :func:`spectrum_circuit` without the QFT pair.

    ``QFT-dagger (QFT D QFT-dagger)^steps`` equals ``D^steps QFT-dagger`` in the
    mirrored layout, so only the QFT-dagger and the phase layers remain.
    """
    q = cudaq.qvector(amps)
    iqft_b(q)
    for _ in range(steps):
        phase_layer_mirrored(q, angles)
        if zero:
            nyquist_phase(q, phi, True)


@cudaq.kernel
def spectrum_circuit_gates(
    n: int,
    angles: list[float],
    phi: float,
    zero: bool,
    swaps: bool,
    steps: int,
    fused: bool,
):
    """The measurement circuits without amplitude loading (resource counting only).

    ``cudaq.estimate_resources`` cannot count a kernel that loads data from an
    amplitude vector, so this wrapper applies the same gates to ``n`` fresh qubits.
    """
    q = cudaq.qvector(n)
    if fused:
        iqft_b(q)
        for _ in range(steps):
            phase_layer_mirrored(q, angles)
            if zero:
                nyquist_phase(q, phi, True)
    else:
        spectral_advection_on(q, angles, phi, zero, swaps, steps)
        iqft_b(q)
