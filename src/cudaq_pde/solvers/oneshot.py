"""Classical bookkeeping for the one-shot advection-diffusion circuits.

No state vectors here.

Equation ``u_t + c u_x = nu u_xx``, periodic, ``N = 2**n`` points, one shot to any
time ``T``. In Fourier space every mode ``m`` (two's complement, ``m = -N/2 ..
N/2-1``) is multiplied by ``exp(-i c kappa m T) * g(m)`` with

    g(m) = exp(-a m**2),    a = nu T kappa**2,    kappa = 2 pi / L.

The circuit is ``QFT-dagger -> advection phases -> block-encoded diag(g) -> QFT``
with the block-encoding ancillas post-selected on ``|0>``. ``g <= 1`` so the diagonal is
a contraction, and the success probability is

    P_succ = sum_m |c_m|^2 g(m)^2 = ||u(T)||^2 / ||u0||^2

(route C divides by ``lambda**2``)

where ``c_m`` are the normalized Fourier amplitudes of ``u0``. Three routes:

``A``  one ancilla, multiplexed RY with ``2**n`` rotations (exact, small ``n``).
``B``  product of ``n(n+1)/2`` contractions on the sign-magnitude register, each a
       controlled RY on an ancilla (explicit ancillas ``B2``, or one ancilla with
       mid-circuit measurement and reset ``B1``). Polynomial in ``n``.
``C``  Gaussian real-space LCU: ``g`` is the Fourier multiplier of a convolution with
       weights ``w_s``; a width register of ``q`` qubits holds the shift ``s``. The
       Gaussian has width ``sigma = sqrt(2 nu T) / h`` grid cells, ``h = L / N``, so at
       a fixed physical ``T`` it grows like ``2**n`` and the cost of preparing the
       weights (``~ 2**q ~ sigma``) is EXPONENTIAL in ``n``. Moderate-``sigma`` route.

Everything here is classical numpy; the kernels are in :mod:`oneshot_kernels`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import hadamard

from cudaq_pde.classical import TWO_PI, Nyquist, _check_nyquist, _check_size
from cudaq_pde.solvers.spectral import phase_angles, shift_angles

ROUTES = ("A", "B", "C")


def diffusion_exponent(nu: float, T: float, L: float = TWO_PI) -> float:
    """``a = nu T (2 pi / L)**2``: the decay of mode ``m`` is ``exp(-a m**2)``."""
    if nu < 0 or T < 0:
        raise ValueError("nu and T must be >= 0")
    return float(nu * T * (TWO_PI / L) ** 2)


def gaussian_width(n: int, nu: float, T: float, L: float = TWO_PI) -> float:
    """``sigma = sqrt(2 nu T) / h`` in grid cells, ``h = L / 2**n``.

    The real-space kernel of ``exp(-a m**2)`` is a Gaussian of this width. At fixed
    ``nu, T, L`` it doubles with every added qubit.
    """
    return float(np.sqrt(2.0 * nu * T) / (L / 2**n))


def modes(n: int) -> NDArray[np.int64]:
    """Integer modes ``m`` in FFT array order (two's complement of the register)."""
    N = 2**n
    return np.where(np.arange(N) < N // 2, np.arange(N), np.arange(N) - N)


def response(n: int, a: float) -> NDArray[np.float64]:
    """``g(m) = exp(-a m**2)`` in FFT array order (Nyquist ``m = -N/2`` included)."""
    return np.exp(-a * modes(n).astype(np.float64) ** 2)


def success_probability(
    u0: NDArray, a: float, kind: str = "exact", lam: float = 1.0
) -> float:
    """``sum |c_m|^2 g(m)^2 = ||u(T)||^2 / ||u0||^2`` from the classical spectrum.

    The advection phases have modulus 1, so only ``g`` enters. ``lam`` is the LCU
    normalization of route C (probability divides by ``lam**2``).
    """
    u0 = np.asarray(u0)
    n = _check_size(u0.size)
    p = np.abs(np.fft.fft(u0)) ** 2
    return float(np.sum(p * response(n, a) ** 2) / np.sum(p) / lam**2)


# ---------------------------------------------------------------------------
# multiplexed RY (Gray-code decomposition), shared by routes A and C
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RySequence:
    """Flat list of ``RY(alpha[i])`` on qubit ``tgt[i]`` then ``CX(ctl[i] -> tgt[i])``.

    ``ctl[i] = -1`` means no CX. The kernels apply exactly this list.
    """

    tgt: list[int]
    alpha: list[float]
    ctl: list[int]

    def reversed(self) -> RySequence:
        """Inverse: reverse order, negate angles (kernel applies CX then RY)."""
        return RySequence(
            self.tgt[::-1], [-x for x in self.alpha[::-1]], self.ctl[::-1]
        )

    def __add__(self, other: RySequence) -> RySequence:
        return RySequence(
            self.tgt + other.tgt, self.alpha + other.alpha, self.ctl + other.ctl
        )

    @property
    def n_ry(self) -> int:
        return len(self.alpha)

    @property
    def n_cx(self) -> int:
        return sum(1 for c in self.ctl if c >= 0)


def multiplexed_ry(
    theta: NDArray[np.float64], ctrl_qubits: list[int], target: int
) -> RySequence:
    """``RY(theta[j])`` on ``target`` for control value ``j``.

    Bit ``b`` of ``j`` is qubit ``ctrl_qubits[b]``.

    Gray-code scheme: ``2**k`` rotations and ``2**k`` CX for ``k`` controls
    (no CX for ``k = 0``). Rotation ``i`` has angle
    ``2**-k * sum_j (-1)**popcount(gray(i) & j) theta[j]``; the CX that follows has the
    control bit where ``gray(i)`` and ``gray(i+1)`` differ.
    """
    k = len(ctrl_qubits)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.size != 2**k:
        raise ValueError(f"need 2**{k} angles, got {theta.size}")
    if k == 0:
        return RySequence([target], [float(theta[0])], [-1])
    size = 2**k
    gray = np.arange(size) ^ (np.arange(size) >> 1)
    alpha = hadamard(size)[gray, :] @ theta / size
    nxt = np.roll(gray, -1)
    bit = np.array([int(d).bit_length() - 1 for d in gray ^ nxt])
    return RySequence(
        [target] * size,
        [float(x) for x in alpha],
        [ctrl_qubits[b] for b in bit],
    )


def prepare_real(v: NDArray[np.float64], qubits: list[int]) -> RySequence:
    """RY/CX tree that maps ``|0..0>`` to the real (signed) vector ``v / ||v||``.

    ``qubits[k]`` holds bit ``k`` of the amplitude index. Level ``l`` rotates qubit
    ``qubits[q-1-l]`` controlled on the ``l`` qubits above it; ``2**q - 1`` RY and
    ``2**q - 2`` CX in total. Signs live in the last level (``atan2``), so negative
    amplitudes cost nothing extra.
    """
    v = np.asarray(v, dtype=np.float64)
    q = len(qubits)
    if v.size != 2**q:
        raise ValueError(f"need 2**{q} amplitudes, got {v.size}")
    seq = RySequence([], [], [])
    for level in range(q):
        block = 2 ** (q - level)  # amplitudes under one prefix
        theta = np.empty(2**level)
        for p in range(2**level):
            part = v[p * block : (p + 1) * block]
            half = block // 2
            if level == q - 1:
                theta[p] = 2.0 * np.arctan2(part[1], part[0])
            else:
                left, right = np.sum(part[:half] ** 2), np.sum(part[half:] ** 2)
                theta[p] = 2.0 * np.arctan2(np.sqrt(right), np.sqrt(left))
        ctrls = [qubits[q - level + j] for j in range(level)]
        seq = seq + multiplexed_ry(theta, ctrls, qubits[q - 1 - level])
    return seq


# ---------------------------------------------------------------------------
# route A
# ---------------------------------------------------------------------------


def route_a_sequence(n: int, a: float) -> RySequence:
    """Multiplexed ``RY(2 arccos g(m))`` on ancilla qubit ``n``.

    Mirrored (swap-free) layout: bit ``b`` of the Fourier index ``m`` lives on qubit
    ``n-1-b``. ``cos(theta/2) = g`` so the ancilla keeps ``|0>`` with amplitude ``g``.
    """
    m = modes(n).astype(np.float64)
    g2 = np.exp(-2.0 * a * m**2)
    theta = 2.0 * np.arctan2(np.sqrt(-np.expm1(-2.0 * a * m**2)), np.sqrt(g2))
    return multiplexed_ry(theta, [n - 1 - b for b in range(n)], n)


# ---------------------------------------------------------------------------
# route B
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contractions:
    """The factors of ``exp(-a m**2)`` on the sign-magnitude register.

    Factor ``i`` is ``RY(theta[i])`` on an ancilla controlled on qubit ``c1[i]`` and,
    if ``c2[i] >= 0``, also on qubit ``c2[i]``; it multiplies the amplitude by
    ``f_i = cos(theta[i]/2) = exp(-a w_i) <= 1`` when the controls are 1.
    """

    c1: list[int]
    c2: list[int]
    theta: list[float]
    weight: list[float]  # w_i, exponent weight of the factor

    @property
    def count(self) -> int:
        return len(self.theta)

    def n_controls(self) -> tuple[int, int]:
        """Numbers of singly and doubly controlled rotations."""
        singles = sum(1 for c in self.c2 if c < 0)
        return singles, self.count - singles


def contractions(n: int, a: float) -> Contractions:
    """Factor ``exp(-a m**2)`` into ``n`` CRY and ``n(n-1)/2`` CCRY contractions.

    Layout (mirrored): the sign bit ``s`` (top Fourier bit, two's-complement weight
    ``-N/2``) is qubit 0, low bit ``b`` is qubit ``n-1-b``. A CX from ``s`` onto every
    low bit maps ``low -> v = low XOR s`` (self-inverse). Then ``|m| = v`` for ``s = 0``
    and ``|m| = v + 1`` for ``s = 1`` (``v = N/2 - 1 - low``; ``low = 0`` is the Nyquist
    mode ``|m| = N/2``), so

        m**2 = v**2 + s (2 v + 1)
             = sum_b 4**b v_b + 2 sum_{b<b'} 2**(b+b') v_b v_b'
               + 2 s sum_b 2**b v_b + s,

    every coefficient positive. Each term is one factor ``exp(-a w)`` with ``w`` its
    coefficient, a contraction. (In two's complement ``m**2`` has negative cross terms
    with the sign bit; those factors would exceed 1 and are not block-encodable.)
    """
    if n < 2:
        raise ValueError(f"need n >= 2, got {n}")
    qv = [n - 1 - b for b in range(n - 1)]  # qubit of v_b
    c1: list[int] = []
    c2: list[int] = []
    w: list[float] = []
    for b in range(n - 1):  # single bits
        c1.append(qv[b])
        c2.append(-1)
        w.append(4.0**b)
    c1.append(0)  # constant term of the sign bit
    c2.append(-1)
    w.append(1.0)
    for b in range(n - 1):  # s * v_b
        c1.append(0)
        c2.append(qv[b])
        w.append(2.0 * 2.0**b)
    for b in range(n - 1):  # v_b v_b'
        for bp in range(b + 1, n - 1):
            c1.append(qv[b])
            c2.append(qv[bp])
            w.append(2.0 * 2.0 ** (b + bp))
    weight = np.array(w)
    theta = 2.0 * np.arctan2(np.sqrt(-np.expm1(-2.0 * a * weight)), np.exp(-a * weight))
    return Contractions(c1, c2, [float(t) for t in theta], [float(x) for x in w])


def contraction_values(n: int, a: float) -> NDArray[np.float64]:
    """``g`` rebuilt from the factors, in FFT order (numpy; test helper).

    Evaluates the product of the ``f_i`` over all register values, applying the CX fold
    and the control pattern exactly as the circuit does.
    """
    fac = contractions(n, a)
    out = np.empty(2**n)
    for j in range(2**n):  # j = Fourier index, bit b = f_b
        f = [(j >> b) & 1 for b in range(n)]
        bit = [0] * n  # qubit values: qubit n-1-b holds f_b
        for b in range(n):
            bit[n - 1 - b] = f[b]
        for q in range(1, n):
            bit[q] ^= bit[0]
        val = 1.0
        for c1, c2, th in zip(fac.c1, fac.c2, fac.theta, strict=True):
            if bit[c1] and (c2 < 0 or bit[c2]):
                val *= np.cos(th / 2.0)
        out[j] = val
    return out


# ---------------------------------------------------------------------------
# route C
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianPlan:
    """Weights ``w_s`` of the real-space convolution and the width register.

    ``g(m) = sum_s w_s exp(-2 pi i m s / N)`` over all ``N`` shifts, ``w`` being the
    inverse DFT of ``g`` (real and even). Only the window ``s in [-2**(q-1), 2**(q-1))``
    is kept; ``dropped`` is the L1 mass left out (an upper bound on the operator error).
    """

    q: int
    weights: list[float]  # w_s for s = -2**(q-1) .. 2**(q-1)-1, as a window of 2**q
    lam: float  # sum |w_s| over the window
    dropped: float
    shift_angles: list[float]  # flat, q blocks of n angles (width bit b)
    prep_r: RySequence  # PREPARE with signs, on the width qubits
    prep_l: RySequence  # PREPARE of |w|^(1/2) (applied as its inverse)

    @property
    def n_negative(self) -> int:
        return sum(1 for w in self.weights if w < 0)


def gaussian_weights(n: int, a: float) -> NDArray[np.float64]:
    """Real-space kernel ``w_s`` (index ``s mod N``): inverse DFT of ``g``."""
    w = np.fft.ifft(response(n, a))
    return w.real


def gaussian_plan(n: int, a: float, tail_tol: float = 1e-13) -> GaussianPlan:
    """Smallest width register whose dropped L1 mass is at most ``tail_tol``."""
    N = 2**n
    w = gaussian_weights(n, a)
    total = float(np.sum(np.abs(w)))
    for q in range(1, n + 1):
        order = np.arange(2**q)
        shifts = np.where(order < 2 ** (q - 1), order, order - 2**q)
        win = w[shifts % N]  # amplitude index = s mod 2**q (two's complement)
        dropped = total - float(np.sum(np.abs(win)))
        if dropped <= tail_tol or q == n:
            break
    lam = float(np.sum(np.abs(win)))
    amp = np.sqrt(np.abs(win) / lam)
    qubits = list(range(q))
    prep_l = prepare_real(amp, qubits)
    prep_r = prepare_real(amp * np.where(win < 0, -1.0, 1.0), qubits)
    unit = shift_angles(n, +1)  # increment: one cell
    weights_b = [2.0**b for b in range(q - 1)] + [-(2.0 ** (q - 1))]
    flat = [float(wb * ang) for wb in weights_b for ang in unit]
    return GaussianPlan(
        q=q,
        weights=[float(x) for x in win],
        lam=lam,
        dropped=dropped,
        shift_angles=flat,
        prep_r=prep_r,
        prep_l=prep_l,
    )


# ---------------------------------------------------------------------------
# shared plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OneShotPlan:
    """Everything the kernels need for one ``(n, c, nu, L, T)``."""

    n: int
    a: float
    sigma: float
    angles: list[float]  # advection phases, "negative" convention
    phi: float  # extra Nyquist phase for the "zero" convention
    zero: bool


def oneshot_plan(
    n: int,
    c: float,
    nu: float,
    L: float = TWO_PI,
    T: float = 1.0,
    nyquist: Nyquist = "zero",
) -> OneShotPlan:
    _check_nyquist(nyquist)
    angles, phi = phase_angles(n, c, L, T)
    return OneShotPlan(
        n=n,
        a=diffusion_exponent(nu, T, L),
        sigma=gaussian_width(n, nu, T, L),
        angles=angles,
        phi=phi,
        zero=nyquist == "zero",
    )
