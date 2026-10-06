# cudaq-pde

Open-source library of quantum PDE solvers built on CUDA-Q (Python API).
Goal: correct, hardware-portable, well-tested reference implementations,
plus honest resource accounting. Not a claim of quantum advantage.

## Scope
This repo's rules come only from this CLAUDE.md. Ignore
/bigdata/rz/ramakr74/AGENTS.md; it belongs to a different project.
Workflow: one branch per checkpoint (e.g. cp1-scaffold), merge to main
after my approval.

## Environment
- CUDA-Q: cudaq>=0.16,<0.17. Python 3.12.4 (for JURECA parity).
- Dev on ROSI (HZDR). Login node: CPU only (target qpp-cpu).
  GPU runs (A100) via srun/sbatch only. Later: JURECA H100, possibly B200.
- Never run GPU jobs on the login node.
- Login node limits: processes are killed after 300 s of CPU time (all threads
  count), and the qpp-cpu simulator spawns OpenMP threads on every core.
  `env/activate_rosi.sh` sets `OMP_NUM_THREADS=1` only when `SLURM_JOB_ID` is
  unset (login node); inside a Slurm job the defaults are kept. Unit tests and
  the resource table run on the login node; anything heavier (large sweeps,
  GPU tests, benchmarks) runs via sbatch from the repo root.

### Activation (ROSI)
```bash
source env/activate_rosi.sh
```
or
```bash
source /bigdata/rz/ramakr74/cudaq_pde/env/activate_rosi.sh
```
Loads python/3.12.4 module, sets PIP_CACHE_DIR, activates venv at `/bigdata/rz/ramakr74/venvs/cudaq-pde`.
Do NOT load any CUDA module; CUDA-Q wheels bring their own runtime.

### Install and test
```bash
pip install -e ".[cu12,dev]"   # or [cu13,dev]; install exactly one CUDA-Q variant
pytest                          # CPU (qpp-cpu); add --run-gpu only on a GPU node
ruff check . && ruff format --check .
```

### Layout
- `src/cudaq_pde/`: `classical.py`, `encoding.py`, `encoding_kernels.py`,
  `metadata.py`, `solvers/` (`spectral.py` driver, `spectral_kernels.py`
  kernels, `spectral_resources.py` gate counts)
- `tests/`, `examples/`, `benchmarks/`, `logs/` (sbatch output, gitignored)
- `env/`: activation script, lock file, one-off environment checks (excluded from ruff)
- Submit sbatch jobs from the repo root, output goes to `logs/%x_%j.out`.
- `benchmarks/spectral_resources.py` prints the resource table;
  `benchmarks/gpu_tests.sbatch` runs `pytest --run-gpu -m gpu` on one A100.

### Verified
- A100 GPU (driver 570.158.01): nvidia fp64 target verified to 1e-12 agreement with CPU.
- See `env/ENVIRONMENT.md` for full details: system, hardware, dependencies, job template.

## Numerical reproducibility (non-negotiable)
- GPU target always: cudaq.set_target("nvidia", option="fp64").
- All randomness seeded and seed exposed as a parameter.
- Every benchmark/result writes metadata: cudaq version, target, GPU name,
  driver, CUDA runtime, Python, numpy/scipy versions, git commit.
- Tests must pass on qpp-cpu (CI has no GPU). GPU tests marked @pytest.mark.gpu.

## Encoding conventions
- n qubits encode N = 2**n grid values via amplitude encoding (normalized).
  Track the norm classically and document it.
- Periodic boundary conditions in the MVP.
- Qubit ordering (measured on CUDA-Q 0.16, tested against numpy in
  `tests/test_encoding_cudaq.py`): grid index j = sum_k b_k 2^k, qubit k holds
  bit b_k. Qubit 0 is the least significant bit, qubit n-1 the most
  significant. `psi[j] = u[j] / ||u||` equals the `cudaq.get_state` layout, so
  no bit reversal is needed. In `cudaq.sample` bitstrings character k is qubit
  k (the string reads LSB first); convert with
  `cudaq_pde.encoding.bitstring_to_index`. All kernels (QFT, phases) must use
  this convention.
- Wavenumbers: `classical.wavenumbers(n, L)` gives k in FFT array order
  (0, 1, ..., N/2-1, -N/2, ..., -1 times 2 pi / L). The Nyquist mode is
  m = -N/2. `evolve_spectral(..., nyquist=...)` fixes the Nyquist convention:
  "zero" (default) uses k = 0 for odd derivatives and k^2 = (pi N / L)^2 for
  diffusion, so real data stays real; "negative" uses k = -N/2, which is what
  a per-qubit phase circuit applies. Both are unitary for nu = 0. See the
  `classical.py` module docstring.

## Spectral advection circuit (solvers/spectral*.py)
- CUDA-Q 0.16 has no QFT; `spectral_kernels.qft_b` is circuit B (H first on
  qubit 0, controlled R1(pi/2^d), no final swaps). Measured against numpy
  (`tests/test_spectral_qft.py`, n = 2..6, all basis inputs, 1e-12):
  `qft_b = F+ R`, where `F+ = sqrt(N) * numpy.fft.ifft` (exp(+2 pi i jk/N)
  / sqrt(N)) and R is the bit reversal. The QFT has the + sign, so it is the
  inverse DFT up to normalization: QFT = `qft_b` after R, QFT-dagger =
  `fft / sqrt(N)`.
- Hence the circuit is QFT-dagger -> phases -> QFT (numpy `fft` is the
  QFT-dagger), with phase exp(-i c (2 pi / L) k t).
- Swap-free is exact, not approximate: U = QFT D QFT-dagger = B R D R B-dagger
  and R D R is D with the bit index reversed, so the default circuit is
  B-dagger, phases on mirrored qubits (Fourier bit b on qubit n-1-b), B.
  `swaps=True` keeps B-dagger, R, D, R, B. Same unitary (tested to 1e-13).
- Phase layer: k is the two's-complement integer of the register (top bit
  weight -2^(n-1)), so exp(-i c kappa t k) is n single-qubit R1 gates with
  theta_b = -c kappa t w_b, kappa = 2 pi / L. This is the "negative" Nyquist
  convention. "zero" adds one (n-1)-controlled R1(-c kappa t N/2) on the
  Nyquist state (X conjugation on the clear qubits). Angles are linear in t,
  so m steps of dt equal one shot of m*dt exactly.
- `spectral.evolve*` read the state with `cudaq.get_state`: SIMULATOR-ONLY.
- Closed-form native gate counts per step (swap-free, "negative"): h = 2n,
  cr1 = n(n-1), r1 = n, total n^2 + 2n, two-qubit gates n(n-1), no swaps.
  `swaps=True` adds 2*floor(n/2) swaps. "zero" adds 2(n-1) X and one gate with
  n-1 controls. Lowered CX (CUDA-Q OpenQASM 2 lowering, not hardware-native):
  2n(n-1) for "negative" (cr1 = 2 CX, swap = 3 CX); "zero" adds 12n-22 CX for
  n >= 3 and n-2 ancilla qubits in that lowering.

## Hardware portability rule
- Solvers intended for QPUs are gate-level @cudaq.kernel code
  (QFT, exp_pauli, rotations, CNOT). No cudaq.evolve / dynamics target in
  solver code paths; dynamics may only be used as an independent check.
- Gate-based targets of interest: IQM, IonQ, AQT. D-Wave and Pasqal analog
  are out of scope for gate kernels.
- Full statevector readout costs exponential shots. Simulator-only
  conveniences (cudaq.get_state) must be clearly marked; provide or plan
  measurable alternatives (observables, Hadamard/swap tests).

## Physics roadmap
1. Linear advection (unitary): spectral solver, QFT -> diagonal phases -> iQFT.
2. Advection-diffusion (non-unitary): variational time stepping
   (Lubasch et al., PRA 101, 010301, 2020 style cost).
3. Burgers: Cole-Hopf to heat equation, then nonlinear variational ansatz.
Classical reference (numpy FFT, exact for linear periodic case) is the
ground truth for every test.

## Code standards
- src/ layout, type hints, numpy-style docstrings, ruff + pytest.
- Small modules, no premature abstraction. Ask before adding dependencies.
- Don't invent CUDA-Q APIs: check the installed version's docs/signatures
  (help(), inspect) when unsure, and say so.
- All `@cudaq.kernel` functions live in dedicated `*_kernels.py` modules.
  Kernel gate names (h, cx, ...) look undefined to ruff, so F821 is ignored
  per file only for `*_kernels.py` via `[tool.ruff.lint.per-file-ignores]`
  in pyproject.toml. Do not ignore F821 anywhere else.
- Writing style in docs/README: no em dashes.
