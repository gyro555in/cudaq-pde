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
- `src/cudaq_pde/`: `classical.py`, `encoding.py`, `metadata.py`, `solvers/`
- `tests/`, `examples/`, `benchmarks/`, `logs/` (sbatch output, gitignored)
- `env/`: activation script, lock file, one-off environment checks (excluded from ruff)
- Submit sbatch jobs from the repo root, output goes to `logs/%x_%j.out`.

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
- Bit/qubit ordering must be documented and tested once against numpy.

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
- Writing style in docs/README: no em dashes.
