# cudaq-pde

Gate-level, hardware-portable reference implementations of quantum PDE solvers
on [CUDA-Q](https://nvidia.github.io/cuda-quantum/) (Python API), with honest
resource accounting. Version 0.1.0.

Two solvers for the 1D periodic equation `u_t + c u_x = nu u_xx` on
`N = 2**n` grid points, encoded in the amplitudes of `n` qubits:

- **Spectral advection** (`nu = 0`, unitary): `QFT-dagger -> n single-qubit phase
  gates -> QFT`, written as `@cudaq.kernel` code from QFT, controlled phases and
  rotations only. One circuit evolves to any time `T` (the angles are linear in
  `T`), and the QFT bit reversals can be dropped exactly.
- **Variational advection-diffusion** (`nu > 0`, non-unitary): explicit-Euler time
  steps, each one a variational fit of a real-amplitude RY/CX ansatz.

The exact classical solutions (`cudaq_pde.classical`) are the ground truth for
every test: `evolve_spectral` for the spectral solver and `evolve_fd_euler`,
which has the same discretization, for the variational one.

## What this is not

- **No quantum advantage claim.** This is a correctness and resource-accounting
  reference, not a speedup.
- **Reading the whole field is exponentially expensive.** Estimating all `N = 2**n`
  amplitudes on a quantum computer needs a number of shots that grows exponentially
  with `n`. The drivers read the state vector with `cudaq.get_state`, which only
  exists on simulators; every function that does so is marked SIMULATOR-ONLY.
  Observables and overlaps (Hadamard or swap tests) are the measurable outputs.
- **The default variational solver evaluates its cost on the simulator** from the
  state vector. A shot-based version that never reads a state vector
  (`cudaq_pde.solvers.measurable`, see "Measurable path") is implemented and tested on
  the simulator with sampled counts; no QPU has run it.
- Scope: 1D, periodic boundaries, real fields in the variational solver, explicit
  Euler (stable only for `nu > 0` and `dt <= min(2 nu / c**2, h**2 / (2 nu))`,
  which the code asserts).
- Verification so far: all CPU tests on the `qpp-cpu` simulator; the spectral solver
  and the qubit-ordering tests also pass on an NVIDIA A100 with the `nvidia` target
  in fp64. The variational solver has CPU tests only.

## Install

Python 3.11 or newer. Install exactly one CUDA-Q variant through an extra:

```bash
pip install -e ".[cu12,dev]"     # NVIDIA driver older than 580
pip install -e ".[cu13,dev]"     # NVIDIA driver 580 or newer
```

The CUDA-Q wheels bring their own CUDA runtime, so no CUDA module or toolkit is
needed; only the driver matters (check with `nvidia-smi`). A newer bundled runtime
runs on an older driver within the same CUDA major version through CUDA minor
version compatibility; for example the 12.9 runtime was verified on a driver that
natively supports 12.8. `[dev]` adds pytest, ruff and matplotlib (used by the
examples). Never mix the `cu12` and `cu13` extras.

On a GPU use `cudaq.set_target("nvidia", option="fp64")`; double precision is
required for the tolerances quoted here.

## Quick start

Spectral advection of a Gaussian pulse, compared with the exact spectral solution:

```python
import numpy as np
import cudaq
from cudaq_pde import classical
from cudaq_pde.solvers import spectral

cudaq.set_target("qpp-cpu")  # on a GPU node: cudaq.set_target("nvidia", option="fp64")
n, c, L, T = 5, 1.0, 2 * np.pi, 1.0
x = classical.grid(n, L)
u0 = np.exp(-0.5 * ((x - np.pi) / 0.3) ** 2)

u = spectral.evolve(u0, c, L, dt=T)  # one circuit for time T
ref = classical.evolve_spectral(u0, T, c, 0.0, L)  # exact spectral solution
print("max error", np.max(np.abs(u - ref)))
```

Variational advection-diffusion, validated against explicit-Euler finite differences:

```python
import numpy as np
import cudaq
from cudaq_pde import classical
from cudaq_pde.solvers import variational

cudaq.set_target("qpp-cpu")
n, c, nu, L, dt, steps = 3, 1.0, 0.4, 2 * np.pi, 0.05, 5
x = classical.grid(n, L)
u0 = 1.0 + 0.5 * np.sin(x) + 0.25 * np.cos(2 * x)

res = variational.evolve_variational(u0, dt, steps, c, nu, L, depth=2, seed=0, tol=1e-6)
ref = classical.evolve_fd_euler(u0, dt, steps, c, nu, L)
fidelity, l2 = variational.compare_to_reference(res, ref)
print("final relative L2 error vs FD-Euler", l2[-1])
print("cost evaluations", int(res.nfev.sum()))
```

Runnable versions that print the errors and save a plot and a metadata JSON to
`results/`:

```bash
python examples/advection_spectral.py --n 6
python examples/advection_diffusion_variational.py --n 3
```

Both examples report the optimization error (variational against FD-Euler) and the
discretization error (FD-Euler against the exact spectral solution) separately.
Tests: `pytest` (CPU) and `pytest --run-gpu -m gpu` on a GPU node.

## Conventions that matter

- **Qubit ordering:** grid index `j = sum_k b_k 2**k`, qubit `k` holds bit `b_k`, so
  qubit 0 is the least significant bit and `psi[j] = u[j] / ||u||` is exactly the
  `cudaq.get_state` layout. In `cudaq.sample` bitstrings character `k` is qubit `k`.
- **QFT:** CUDA-Q 0.16 has none, so it is written here. The QFT has the + sign, it is
  `sqrt(N) * numpy.fft.ifft`, so numpy's `fft` is the QFT-dagger and the evolution
  is `QFT-dagger -> phases -> QFT`. Measured against numpy in the tests.
- **Nyquist mode:** its wavenumber is a convention. `"zero"` (default) sets `k = 0`
  for odd derivatives and keeps real data real; `"negative"` uses `k = -N/2`, which
  a per-qubit phase circuit applies directly. Both are unitary for `nu = 0` and
  both are implemented in `classical.evolve_spectral` and in the circuit.
- **Norm:** amplitude encoding keeps only the direction of `u`; the norm is tracked
  classically (signed in the variational solver, where `psi` and `-psi` are the same
  state).

## Resource summary

Spectral advection, one step (one circuit for any time), counts from
`cudaq.estimate_resources` on the real kernel and, for CX, CUDA-Q 0.16's OpenQASM
lowering. Lowered CX counts depend on that compiler lowering and are not
hardware-native; native counts for a given device come with the hardware runs.

- Swap-free circuit, `"negative"` convention: `n**2 + 2n` gates (`2n` Hadamards,
  `n(n-1)` controlled phases, `n` single-qubit phases), `n(n-1)` two-qubit gates,
  `2n(n-1)` CX after lowering, no swaps.
- Keeping the QFT bit reversals (`swaps=True`) adds `2*floor(n/2)` swaps (3 CX each).
  Dropping them is exact: `QFT D QFT-dagger = B R D R B-dagger`, and conjugating the
  diagonal `D` by the bit reversal `R` just mirrors the qubit index of the phases.
- The `"zero"` Nyquist convention adds `2(n-1)` X gates and one `(n-1)`-controlled
  phase on the single state `|N/2>`: about `12n - 22` extra CX for `n >= 3`, `n - 2`
  ancilla qubits in CUDA-Q's lowering, and one more logical layer.

| n | gates | two-qubit | lowered CX | "zero" extra gates | "zero" extra CX | "zero" extra ancillas |
|---|-------|-----------|------------|--------------------|-----------------|-----------------------|
| 4 | 24 | 12 | 24 | 7 | 26 | 2 |
| 6 | 48 | 30 | 60 | 11 | 50 | 4 |
| 8 | 80 | 56 | 112 | 15 | 74 | 6 |
| 10 | 120 | 90 | 180 | 19 | 98 | 8 |

(`tests/test_readme.py` recomputes this table from the code.)

Variational solver, hardware version of the cost `C = 1 - <psi|b>**2 / ||b||**2`:
`I + dt L` is a combination of the unitary cyclic shifts `decrement`, `increment` and
the identity (`np.roll(u, -1)` and `np.roll(u, +1)`), so each overlap
`<psi(theta)| V |psi_prev>` with `V` in `{I, decrement, increment}` is one
Hadamard-test circuit (all amplitudes are real, so one circuit per overlap).

- **3 distinct overlap circuits per cost evaluation.**
- 2 further circuits per step (`<decrement>` and `<decrement**2>` on the previous
  state) for
  `||b||**2`, independent of `theta`.
- A finite-difference gradient costs `P + 1` cost evaluations, `P = n (depth + 1)`.
- The shifts are the spectral circuit above with the `"negative"` convention.

## Measurable path (shots only, no state vector)

`cudaq_pde.solvers.measurable` and `spectral_measure` use only `cudaq.sample` counts;
a test enforces that no hardware-path module calls `cudaq.get_state`, and that
`cudaq.sample` has a single call site (`solvers/sampling.py`).

- **Variational cost.** The target `b = (I + dt L) psi_prev` is prepared by a linear
  combination of unitaries (LCU) with two ancillas, `A(theta)^dagger` is applied and
  everything is sampled: `C = 1 - P(sys = 0 | anc = 00)` and the norm update
  `|norm| lambda sqrt(P_joint)` come from the same counts. One circuit per cost
  evaluation, no controlled ansatz. Gradients are exact parameter-shift gradients
  (`2P` circuits); the optimizer is heavy-ball descent.
- **Compared with a Hadamard test** (3 overlap circuits per evaluation, controlled
  ansatz, plus 2 per-step circuits): at n = 3, depth 2 the LCU circuit is 36 lowered
  CX against 288 for the three Hadamard circuits (CUDA-Q lowering, not
  hardware-native), and at equal total shots its estimator has about 3.3 times lower
  standard deviation, which goes to zero at the optimum while the Hadamard test's does
  not (numpy Monte Carlo of the estimators).
- **Shot noise** (n = 3, 5 steps, final infidelity to explicit-Euler FD, mean of 3
  seeds): about 1.6e-2, 1.2e-3 and 1.5e-4 at 100, 1,000 and 10,000 shots per circuit,
  with 1081 circuits (1.08e5 to 1.08e7 shots) per time step. The full sweep is
  `benchmarks/shot_study.sbatch`.
- **Power spectrum.** After the evolution, a swap-free QFT-dagger and sampling give
  `P[k] = |u_hat_k|^2 / (N ||u||^2)`. The sampled bitstring index is the bit-reversed
  FFT index. The error matches the multinomial prediction and scales as
  shots^-0.51 for n = 3 to 8.

## Roadmap

- Burgers equation via the Cole-Hopf transform to the heat equation, then a
  nonlinear variational ansatz.
- Shot-noise and parameter-shift study of the variational cost.
- Hardware runs (IQM, AQT) through CUDA-Q targets, with native gate counts.

## Citation and license

See `CITATION.cff`. A license has not been chosen yet and must be set before the
first release.
