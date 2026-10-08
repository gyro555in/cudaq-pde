"""Sampled Cole-Hopf heat step for Burgers (hardware path: shots, no state vector).

``phi0 = exp(-(1/2nu) int w0)`` (zero-mean ``w0``, see :mod:`cudaq_pde.burgers`) is
loaded by gates, the CP7 one-shot heat evolution runs on the chosen route, and the
system and ancillas are sampled. Shots with all ancillas 0 are accepted; for a real
positive state

    c_j / accepted  ->  p_j = phi_j(T)^2 / ||phi(T)||^2,

and with ``S`` total shots the joint probability of "accepted and bin j" is
``q_j = P_succ p_j`` (route C: ``P_succ`` divided by ``lambda**2``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cudaq_pde import encoding
from cudaq_pde.classical import TWO_PI
from cudaq_pde.solvers import burgers_kernels as bk
from cudaq_pde.solvers import oneshot
from cudaq_pde.solvers.sampling import Sampler
from cudaq_pde.solvers.spectral_measure import bit_reverse_index

ROUTES = ("A", "B2", "C")


@dataclass(frozen=True)
class HeatCall:
    """Kernel, arguments and bookkeeping of one sampled heat step."""

    kernel: object
    args: tuple
    n: int
    n_anc: int
    lam: float  # LCU normalization (1 for routes A and B)


def heat_call(
    phi0: NDArray[np.floating],
    nu: float,
    T: float,
    L: float = TWO_PI,
    route: str = "A",
    fourier: bool = False,
    tail_tol: float = 1e-13,
) -> HeatCall:
    """Build the measured heat-step circuit for ``phi0`` (real, positive)."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    psi, _ = encoding.encode(np.asarray(phi0, dtype=np.float64))
    n = encoding.num_qubits(psi.size)
    a = oneshot.diffusion_exponent(nu, T, L)
    ld = oneshot.prepare_real(psi.real, list(range(n)))
    if route == "A":
        seq = oneshot.route_a_sequence(n, a)
        args = (n, fourier, ld.tgt, ld.alpha, ld.ctl, seq.alpha, seq.ctl)
        return HeatCall(bk.heat_a_measured, args, n, 1, 1.0)
    if route == "B2":
        fac = oneshot.contractions(n, a)
        args = (n, fac.count, fourier, ld.tgt, ld.alpha, ld.ctl)
        args += (fac.c1, fac.c2, fac.theta)
        return HeatCall(bk.heat_b2_measured, args, n, fac.count, 1.0)
    g = oneshot.gaussian_plan(n, a, tail_tol)
    lr = g.prep_l.reversed()
    args = (n, g.q, fourier, ld.tgt, ld.alpha, ld.ctl, g.shift_angles)
    args += (g.prep_r.tgt, g.prep_r.alpha, g.prep_r.ctl, lr.tgt, lr.alpha, lr.ctl)
    return HeatCall(bk.heat_c_measured, args, n, g.q, g.lam)


@dataclass(frozen=True)
class HeatSample:
    """Counts of one sampled heat step."""

    bin_counts: NDArray[np.int64]  # accepted shots per REGISTER index (see below)
    accepted: int
    shots: int
    n: int
    n_anc: int
    lam: float
    fourier: bool

    @property
    def p_succ_hat(self) -> float:
        """Sampled acceptance ``accepted / shots`` (route C: ``P_succ / lambda^2``)."""
        return self.accepted / self.shots

    @property
    def grid_counts(self) -> NDArray[np.int64]:
        """Accepted counts per grid index ``j`` (position-space readout)."""
        if self.fourier:
            raise ValueError("this sample is a Fourier-basis readout")
        return self.bin_counts

    def fft_counts(self) -> NDArray[np.int64]:
        """Accepted counts per numpy FFT index (the register index is bit-reversed)."""
        if not self.fourier:
            raise ValueError("this sample is a position-space readout")
        out = np.zeros_like(self.bin_counts)
        for r, c in enumerate(self.bin_counts):
            out[bit_reverse_index(r, self.n)] += c
        return out


def sample_heat(
    phi0: NDArray[np.floating],
    nu: float,
    T: float,
    L: float = TWO_PI,
    route: str = "A",
    shots: int = 10_000,
    seed: int = 0,
    fourier: bool = False,
    tail_tol: float = 1e-13,
    sampler: Sampler | None = None,
) -> HeatSample:
    """Sample the heat step; ancilla characters all ``0`` mean accepted."""
    call = heat_call(phi0, nu, T, L, route, fourier, tail_tol)
    sampler = sampler or Sampler(base_seed=seed)
    counts = sampler.counts(call.kernel, *call.args, shots=shots)
    n = call.n
    bins = np.zeros(2**n, dtype=np.int64)
    accepted = 0
    for bits, c in counts.items():
        if set(bits[n:]) <= {"0"}:
            bins[encoding.bitstring_to_index(bits[:n])] += c
            accepted += c
    return HeatSample(bins, accepted, shots, n, call.n_anc, call.lam, fourier)


def expected_p_succ(
    phi0: NDArray[np.floating],
    nu: float,
    T: float,
    L: float = TWO_PI,
    route: str = "A",
    tail_tol: float = 1e-13,
) -> float:
    """Exact acceptance probability of the route (classical formula)."""
    a = oneshot.diffusion_exponent(nu, T, L)
    lam = 1.0
    if route == "C":
        lam = oneshot.gaussian_plan(encoding.num_qubits(np.size(phi0)), a, tail_tol).lam
    return oneshot.success_probability(phi0, a, lam=lam)
