# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - unreleased

The release date is to be filled in when the version is tagged.

### Added
- Package scaffold with a src layout, optional CUDA-Q extras (`cu12`, `cu13`), a
  `gpu` pytest marker skipped unless `--run-gpu` is given, CPU-only CI for Python
  3.11 and 3.12, and run metadata collection (`cudaq_pde.metadata`) recording the
  CUDA-Q version, target, GPU, driver, CUDA runtime, package versions and git
  commit.
- Classical references: `evolve_spectral` with an explicit Nyquist convention
  (`"zero"` keeps real data real, `"negative"` is what a per-qubit phase circuit
  applies; both unitary for `nu = 0`), and `evolve_fd_euler`, explicit Euler on
  the periodic central-difference operator with an asserted stability limit.
- Amplitude encoding with the measured qubit ordering (qubit 0 is the least
  significant bit), checked against numpy.
- Gate-level spectral advection: a QFT written for CUDA-Q 0.16 (which has none),
  with its sign and bit order measured against `numpy.fft`; a phase layer made of
  `n` single-qubit gates on the two's-complement wavenumber; exact removal of the
  QFT bit reversals; fast-forwarding to any time with one circuit; a resource
  report with native counts and CUDA-Q's OpenQASM lowering, including the cost of
  the `"zero"` Nyquist convention.
- Variational advection-diffusion: real-amplitude RY/CX ansatz, per-step fit of
  `(I + dt L)` applied to the previous state, signed classical norm tracking,
  seeded scipy optimizers (L-BFGS-B by default, COBYLA), per-step cost-evaluation
  counts, and a simulator-only cost whose Hadamard-test hardware version is
  documented.
- Examples (`examples/advection_spectral.py`,
  `examples/advection_diffusion_variational.py`), benchmark scripts for the
  resource table and variational accuracy, a GPU test job script, and a README
  whose snippets and resource table are checked by tests.
- Environment reproducibility files and login-node helpers (`env/`).

### Known limitations
- One dimension, periodic boundaries only.
- Explicit Euler needs `nu > 0`; pure advection is handled by the spectral solver.
- The variational cost is evaluated on the simulator; the hardware version is
  described, not run. The variational solver has CPU tests only.
- Lowered CX counts are CUDA-Q's compiler lowering, not hardware-native counts.
- No license has been chosen yet.
