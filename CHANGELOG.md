# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Cyclic shift kernels `increment` / `decrement` (QFT, multi-controlled X, and
  explicit-ancilla forms) and LCU preparation of `(I + dt L)|psi>` with
  post-selection, with resource tables (`benchmarks/measurable_resources.py`). The
  QFT form is built from the spectral phase-layer kernels (a shift is advection by
  one grid cell, cross-tested against the spectral circuit).
- Measurable variational step: LCU cost circuit, sampled cost and norm update,
  parameter-shift gradients, heavy-ball descent, a single `cudaq.sample` wrapper, a
  comparison with the Hadamard-test cost (gate counts and estimator statistics), and
  a shot-noise study script with a CPU sbatch (`benchmarks/shot_study.*`).
- Measurable power spectrum of the evolved field with the bit-reversed ordering
  handled, a fused variant without the cancelling QFT pair, and its shot-noise
  scaling (`benchmarks/spectrum_shots.py`).
- A structural test that only documented simulator modules call `cudaq.get_state`.
- A `slow` pytest marker (skipped unless `--run-slow`) for tests over about 1 s; CI and
  the GPU sbatch pass it. The default login-node run drops from 125 s to about 48 s of CPU after the one-shot tests.

- One-shot advection-diffusion, `QFT-dagger -> advection phases -> block-encoded
  diag(exp(-a m**2)) -> QFT` with post-selected ancillas, in three routes: multiplexed RY
  (exact, small n), a product of `n` CRY and `n(n-1)/2` CCRY contractions on the
  sign-magnitude register (explicit ancillas, or one ancilla with mid-circuit `mz` and
  `reset` through `cudaq.run`), and a Gaussian real-space LCU (exponential in n at fixed
  physical time, documented). Tests against `classical.evolve_spectral` to 1e-12,
  success probability, sampled success probability for the mid-circuit route, negative
  controls, resource accounting (native, lowered, constructed) and
  `benchmarks/oneshot_resources.py`.
- `sampling.run_records`: the single `cudaq.run` call site (per-shot records of
  mid-circuit measurements).

- `pytest --shard K/N` and `env/pytest_login_all.sh`: the suite runs in shards on the
  login node so that no process exceeds about 150 s of CPU; `env/pytest_login.sh` refuses
  `--run-slow` without `--shard`.
- `benchmarks/mid_circuit_probe.py` and `.sbatch`: mid-circuit measurement and reset
  probe for any target (also with `--emulate`), run on an A100 (`nvidia`, fp64).

### Fixed
- `spectral_resources.qasm_stats` split OpenQASM 2 text by line, but CUDA-Q 0.16 writes
  `reset q[3];cx ...` on one line, so the gate after every reset was not counted. It now
  splits statements on `;`.

### Changed
- Angles and infidelities between nearly equal states are computed stably
  (`cudaq_pde.metrics`: `atan2` of the orthogonal residual) instead of `1 - F`, in the
  variational cost, the reference comparison, the numpy ansatz cost, the cost statistics
  and the report scripts. `variational.compare_to_reference` (which returned a fidelity)
  is replaced by `reference_errors`, returning `(angle, infidelity, l2)`. The variational
  cost near the optimum is now accurate down to `C ~ 1e-30` instead of about 1e-16.
- One vocabulary for the cyclic shifts everywhere: `classical.increment` and
  `classical.decrement` (defined as `np.roll(u, +1)` and `np.roll(u, -1)`) replace
  the `S+` / `S-` notation in code and docs. `fd_euler_coefficients` returns the
  same numbers in the same order, now named `(a0, a_dec, a_inc)`.

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
