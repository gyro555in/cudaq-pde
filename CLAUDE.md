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
- CUDA-Q runtime threads spin across all cores even with `OMP_NUM_THREADS=1`
  (measured: 3.8 s CPU for 0.5 s of work). On the login node run tests with
  `env/pytest_login.sh`, which pins to one core so CPU time equals wall time
  (0.4 s CPU for the same work). It refuses to run inside a Slurm job; there use
  plain `pytest`. Pin other login-node scripts the same way with `taskset -c 0`.

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
env/pytest_login.sh             # login node (pinned), slow tests skipped: ~50 s CPU
env/pytest_login_all.sh [N]     # login node, everything except GPU in N shards (default 3)
env/pytest_login.sh --run-slow --shard K/N   # one shard by hand; --run-slow alone is refused
pytest --run-slow               # inside a Slurm job (plain pytest); CI also uses it
pytest --run-gpu --run-slow -m gpu   # GPU tests, only on a GPU node via sbatch
ruff check . && ruff format --check .
```

### Layout
- `src/cudaq_pde/`: `classical.py`, `encoding.py`, `encoding_kernels.py`,
  `metadata.py`, `solvers/` (`spectral*.py`: gate-level advection;
  `variational.py` driver and `variational_kernels.py` ansatz: advection-diffusion)
  (`oneshot*.py`: one-shot advection-diffusion, three block-encoding routes)
  and `burgers.py` (classical Burgers references) with `solvers/burgers_*.py` (Cole-Hopf
  heat step, estimator, studies)
- `tests/`, `examples/`, `benchmarks/`, `logs/` (sbatch output, gitignored),
  `results/` (example plots and JSON, gitignored)
- Release files: `README.md` (snippets and resource table are checked by
  `tests/test_readme.py`), `CHANGELOG.md`, `CITATION.cff`; the version in
  pyproject, CITATION and CHANGELOG must agree (`tests/test_release_metadata.py`).
  No license is chosen yet, and date, DOI and ORCID are deliberately not set.
- Examples (`examples/advection_spectral.py`,
  `examples/advection_diffusion_variational.py`) write a PNG and a metadata JSON
  to `results/`. On the login node run them pinned: `taskset -c 0 python examples/<file>.py`.
- `env/`: activation script, lock file, one-off environment checks (excluded from ruff)
- Submit sbatch jobs from the repo root, output goes to `logs/%x_%j.out`.
- `docs/reproduce_paper.md`: command and output path of every paper table and figure
  (tested); `benchmarks/plots.py` draws the figures from the stored JSON.
- `benchmarks/spectral_resources.py` prints the resource table;
  `benchmarks/variational_report.py` prints the variational accuracy tables;
  `benchmarks/gpu_tests.sbatch` runs `pytest --run-gpu -m gpu` on one A100.
- `benchmarks/regenerate_paper.sh` (login node: measured-fast steps; `regenerate_paper_cpu.sbatch`:
  all) regenerates the CPU outputs from a clean `origin/main` and runs
  `benchmarks/check_provenance.py`; `benchmarks/gpu_release.sbatch` is the GPU counterpart
  (`--nodelist` picks the 40 GB or 80 GB card, see `docs/reproduce_paper.md`).

### Verified
- A100 GPU (driver 570.158.01): nvidia fp64 target verified to 1e-12 agreement with CPU.
- See `env/ENVIRONMENT.md` for full details: system, hardware, dependencies, job template.

## Test markers
- `gpu`: needs a GPU, skipped unless `--run-gpu`. `slow`: a test (or a group sharing
  cached results) taking more than about 1 s, skipped unless `--run-slow`. Mark new tests
  slow when they exceed ~1.5 s; CI and sbatch runs pass `--run-slow`.
- Login-node budget: NO single invocation may exceed about 150 s of CPU (the node kills
  at 300 s). `--shard K/N` (conftest: tests sorted by node id, dealt round robin) splits
  a run; `env/pytest_login.sh` refuses `--run-slow` without `--shard`, and
  `env/pytest_login_all.sh [N]` runs every shard in its own process and prints each
  shard's CPU seconds. Measured after CP8 with N = 3: 102 s, 95 s, 102 s (1530 tests,
  about 300 s in total including three start-ups); the default run alone is about 60 s.
  Raise N (e.g. to 4) when a shard nears 130 s. Do not run `--run-slow` unsharded in one process.
  `tests/test_markers.py` tests the mechanism with the repository's real `conftest.py`.

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

## Variational advection-diffusion (solvers/variational*.py)
- Shift names (one vocabulary everywhere, defined by numpy and pinned by
  `tests/test_shift_definitions.py`): `classical.increment(u) = np.roll(u, +1)`,
  i.e. `(increment u)_j = u_{j-1}`, on basis states `|j> -> |j+1>`;
  `classical.decrement(u) = np.roll(u, -1)`, `(decrement u)_j = u_{j+1}`,
  `|j> -> |j-1>`. The CUDA-Q kernels in `shift_kernels.py` use the same names and
  are tested against these functions. Do not introduce S+/S- again.
- Reference: `classical.evolve_fd_euler` is explicit Euler on the periodic
  central-difference operator, `I + dt L = a0 I + a_dec * decrement + a_inc *
  increment` with `a0 = 1-2r`, `a_dec = r-s` (weight of `u_{j+1}`),
  `a_inc = r+s` (weight of `u_{j-1}`), `r = nu dt/h^2`,
  `s = c dt/(2h)`; `fd_euler_coefficients` returns `(a0, a_dec, a_inc)`. The variational solver is validated against THIS (isolates
  optimization error); FD-Euler vs `evolve_spectral` is the discretization error
  and is reported separately.
- Stability is asserted (`check_fd_euler_stable`, exact discrete amplification):
  `dt <= min(2 nu/c^2, h^2/(2 nu))` is a sufficient closed form. `nu = 0` has no
  stable explicit step, so CP4 needs `nu > 0`.
- Per step `b = (I + dt L) psi_prev` (unnormalized) and
  `C(theta) = 1 - <psi|b>^2/||b||^2`, the infidelity to `b/||b||`. Norm tracking:
  `norm_new = norm_prev * <psi(theta*)|b>`. The norm is SIGNED: C is invariant
  under `psi -> -psi` (global phase) and the optimizer may land on either sign
  (observed negative for n = 4), so the sign is part of the classical scalar and
  `norm * psi` is the correct field.
- Ansatz: RY layer then `depth` x (CX ladder, RY layer), `n(depth+1)` angles, real
  amplitudes. Needed depth for the smooth test field: n = 3 depth 2, n = 4
  depth 3 (observed infidelity 1e-12 and 1e-9; depth 1 stalls at 3e-3 and 8e-3).
- Optimizer: L-BFGS-B with finite-difference gradients (default; `tol` is the
  projected-gradient tolerance, reaching `C ~ tol^2`, floor at the 1e-8 finite
  difference step) or COBYLA (hits `maxiter` and stalls near 1e-4 here, so much
  weaker). Seeded init, warm start, deterministic. `nfev` per step is reported
  (n = 4: ~800 to 1500 per step, ~2800 for the initial fit).
- `variational.simulator_cost_and_state` is the only `cudaq.get_state` call
  (SIMULATOR-ONLY). Hardware version: Hadamard tests on `U_theta^dagger V U_prev`
  with `V in {I, decrement, increment}`: 3 distinct overlap circuits per cost
  evaluation, plus 2 per step (`<decrement>`, `<decrement^2>`) for `||b||^2`; a finite-difference gradient costs
  `P + 1` evaluations. The shifts are the CP3 spectral circuit with the "negative"
  Nyquist convention.
- Global error obeys `phi_s <= arcsin(kappa sin phi_{s-1}) + arcsin(sqrt(C_s))`
  with `kappa = max|g|/min|g|` of `I + dt L`. kappa is not near 1 (1.15 for
  n = 3, 2.08 for n = 4: the highest mode is damped most), so the bound is
  valid but loose.

## Shifts and LCU target preparation (solvers/shift_kernels.py, lcu*.py)
- `increment_*` / `decrement_*` kernels in three forms, tested on every basis state
  for n = 2..6: `qft` (QFT sandwich with the "negative" convention, no ancilla,
  `2n(n-1)` lowered CX), `mcx` (multi-controlled X cascade, no source ancilla but the
  compiler adds some from n = 4), `ladder` (`n-2` explicit ancillas, Toffoli and CX
  only, `13n-25` lowered CX).
- CUDA-Q 0.16 kernels cannot mix one qubit and a qview as control operands, and qviews
  are sliced with Python syntax (`q[a:b]`), not `.slice`. An ancilla-controlled MCX
  cascade therefore needs a Toffoli ladder with `n-1` work ancillas.
- LCU: `I + dt L = a0 I + a_dec * decrement + a_inc * increment` with 2 ancilla qubits
  (slot `00` identity, `a0 = 1` decrement, `a1 = 1` increment, slot `11` unused);
  PREPARE = RY + controlled RY, signs of negative coefficients are a Z on the matching
  ancilla (a_dec can be negative while explicit Euler is stable), SELECT = QFT sandwich
  (`QFT (c-D_dec c-D_inc) QFT-dagger`, `n^2 + n` cr1) or the Toffoli ladder, post-select
  ancillas on `|00>`. Success probability `||b||^2 / lambda^2`, `lambda = sum |c_i|`
  (equals 1 when all coefficients are >= 0). Verified against `evolve_fd_euler` to 1e-12.
- LCU block lowered CX (`benchmarks/measurable_resources.py`): QFT SELECT
  `2(n^2+n)+4`, ladder SELECT `24(n-1)+2n+4` with `n-1` extra ancillas; the ladder only
  wins from n = 11.
- Hardware-path modules never call `cudaq.get_state`; statevector checks are in tests
  and labelled STATEVECTOR CHECK.

## Measurable path (solvers/measurable*.py, sampling.py, spectral_measure.py)
- Rules enforced by `tests/test_hardware_path_purity.py` (AST scan): hardware-path
  modules never call `cudaq.get_state`; only `variational.py`, `spectral.py` and
  `oneshot_sim.py` do (SIMULATOR-ONLY); `cudaq.sample` is called only in
  `sampling.sample_counts` and `cudaq.run` only in `sampling.run_records` (one call site
  each), which own seeding (`Sampler`: `base_seed + circuit_index`). Statevector checks
  live in tests and are labelled STATEVECTOR CHECK.
- QFT shifts and the LCU SELECT reuse the CP3 `phase_layer_mirrored` (and its
  controlled twin in `spectral_kernels.py`); angles come from
  `spectral.shift_angles` (a shift is advection by one cell, c*dt = +-L/N, "negative"
  Nyquist). `tests/test_shift_spectral_crosscheck.py` pins this to the spectral circuit.
- Cost circuit (`measurable_kernels.lcu_cost_circuit`): `A(theta_prev)`, LCU block,
  `A(theta)^dagger`, sample all qubits. `P_joint = P(sys=0, anc=00) = <psi|b>^2/lambda^2`,
  `P_succ = P(anc=00) = ||b||^2/lambda^2`, `C = 1 - P_joint/P_succ`,
  `|norm_new| = |norm_prev| lambda sqrt(P_joint)` (sign = unobservable global phase).
- Compared with the Hadamard test (controlled ansatz, 3 overlap circuits + 2 per step):
  n = 3, depth 2: 36 vs 288 lowered CX per cost evaluation; ~3.3x lower estimator std at
  equal total shots; std -> 0 at the optimum (numpy Monte Carlo). LCU is recommended
  and implemented.
- Gradients: exact parameter shift `[C(theta_i + pi/2) - C(theta_i - pi/2)]/2`, checked
  against finite differences to 1e-8 and for unbiasedness under sampling. Optimizer:
  heavy-ball descent (lr 1, momentum 0.8, 60 iterations, rate decays to 0.1); Adam
  reached only 3e-6 on exact gradients, heavy ball 4e-11. Circuits per step
  `iterations * 2P + 1`.
- Shot study (`benchmarks/shot_study.py`, n = 3, depth 2, 5 steps): final infidelity to
  FD-Euler 1.6e-2 / 1.2e-3 / 1.5e-4 at 1e2 / 1e3 / 1e4 shots (about 1/shots), 1081
  circuits per step. Full sweep via `benchmarks/shot_study.sbatch` (CPU partition, not
  submitted without approval; ~26 min for 10 seeds).
- Power spectrum: `iqft_b = R F-`, so register index `r` is the bit-reversed FFT index,
  `k_fft = rev(r)`; `P[k] = |u_hat_k|^2/(N ||u||^2)`; advection leaves it invariant;
  error follows `sqrt((1 - sum p^2)/N)` and scales as shots^-0.51 (n = 3..8).

## One-shot advection-diffusion (solvers/oneshot*.py)
- Circuit `QFT-dagger -> advection phases (CP3) -> block-encoded diag(g) -> QFT`,
  `g(m) = exp(-a m^2)`, `a = nu T (2 pi / L)^2`, ancillas post-selected on 0. One circuit
  for any `T`, no explicit-Euler stability limit. `g` is even in `m`, so only the
  advection part sees the Nyquist convention. `P_succ = sum |c_m|^2 g(m)^2 =
  ||u(T)||^2 / ||u0||^2` (route C: divided by `lambda^2`). All routes match
  `classical.evolve_spectral` (observed ~1e-15, asserted 1e-12), both conventions.
- `sigma = sqrt(2 nu T) / h` grid cells with `h = L / 2**n` is the width of the
  real-space Gaussian, `a = 2 pi^2 sigma^2 / N^2` (tested). At fixed physical `T`,
  `sigma` doubles with every qubit.
- Files: `oneshot.py` (classical plans, no state vectors), `oneshot_kernels.py`,
  `oneshot_sim.py` (SIMULATOR-ONLY post-selection drivers), `oneshot_resources.py`.
  The system register keeps the swap-free mirrored layout (Fourier bit b on qubit
  n-1-b); ancillas are separate qviews, never one padded vector.
- Route A: one ancilla, multiplexed `RY(2 arccos g)` (Gray code, `2**n` RY + `2**n` CX).
  Exact reference for small n; gate count exponential.
- Route B: sign bit = qubit 0; a CX from it onto the low bits (self-inverse) gives
  `v = low XOR s`, and `m^2 = v^2 + s (2 v + 1)` has only positive coefficients, so
  `g` is a product of `n` CRY and `n(n-1)/2` CCRY contractions (two's complement has
  negative cross terms with the sign bit, factors above 1, not block-encodable; tested).
  A CCRY is written as 3 CRY + 2 CX (8 lowered CX): CUDA-Q 0.16's own two-control RY
  lowers to 14 CX plus a compiler ancilla (measured). B2 = `n(n+1)/2` explicit
  ancillas; B1 = one ancilla with `mz` then `reset` after each factor, run through
  `cudaq.run` (`sampling.run_records`); the factors commute, so the all-zero record has
  the same probability (tested against B2 and the exact distribution).
- Mid-circuit `mz` + `reset` status (report, do not guess; `benchmarks/mid_circuit_probe.py`):
  works on qpp-cpu and on the A100 (nvidia fp64, job 607242: joint statistic, B1, B2 within
  4 sigma, routes A/B/C vs reference 1.6e-15) through `cudaq.run` on a `-> list[bool]`
  kernel. Only `cudaq.run` gives the joint record: `cudaq.sample` with named registers
  keeps only the final register, `sample(..., explicit_measurements=True)` returned
  impossible correlations (~8% of shots), and 0.16 refuses `sample` for kernels that
  branch on a measurement ("use cudaq.run"). Emulated remote targets
  (`set_target(name, emulate=True)`, documented for ionq and quantinuum): `cudaq.run` is
  "not yet supported on this target" for both, so B1's joint record is unavailable there;
  B2 compiles and is identical to qpp-cpu, B1 compiles as a plain `sample` kernel
  (`oneshot_b1_sample_loaded`, counts identical, no joint statistic). IQM emulation
  needs the device architecture from the server URL (no server, no mock in the wheel):
  not tested. Emulation proves compilation and a local run, not hardware support. IonQ's
  target emits `qir-base` and removes measurements; Quantinuum's emits adaptive QIR. AQT
  is not a 0.16 target. `cudaq.translate` accepts mid-circuit kernels as OpenQASM 2 and
  `qir-adaptive`; `qir-base` fails. Keep B2 as the fallback.
- OpenQASM round trip (`benchmarks/qiskit_roundtrip.py`, optional extra `aqt` = Qiskit
  only, `qiskit-aqt` NOT installed): spectral, A, B2 and B1 at n = 3 load in Qiskit 2.5.2
  with equal gate and CX counts; statevectors agree to ~1e-7 because CUDA-Q prints
  OpenQASM angles with `%e` (7 digits); B1's measure+reset pairs survive load and
  dump/load. AQT backend acceptance of mid-circuit reset is not checked.
- Route C: the multiplier `g` is the DFT of real weights `w_s` (inverse DFT of `g`,
  signed). Window `s in [-2**(q-1), 2**(q-1))`, `q` minimal with dropped L1 mass at most
  1e-13; LCU `A_L^dagger SELECT A_R` with a signed RY-tree `A_R` and an unsigned `A_L`;
  the controlled shifts are phase layers inside the SAME QFT sandwich as the advection
  (`q n` cr1). `lambda = sum |w_s|`. Cost of the tree `~ 2**q ~ sigma`: EXPONENTIAL in n
  at fixed physical `T`; for `sigma < ~2.5` the Nyquist cut leaves a ripple and the
  window is the whole grid (`q = n`). Moderate-`sigma` route only.
- Counts (`benchmarks/oneshot_resources.py`, lowered CX, "negative"; every one measured
  from the OpenQASM lowering of a builder mirror and equal to the construction): n = 10:
  A 1204, B 578 (55 ancillas for B2, 1 for B1 with 55 mid-circuit measurements), C at
  `sigma = 4` 424 (q = 6). B beats A from n = 8. `"zero"` adds `12n - 22` CX.
- `qasm_stats` splits statements on `;` (CUDA-Q writes `reset q[3];ry(...) q[3];` on one
  line; splitting on lines silently dropped the gate after each reset).

## Burgers via Cole-Hopf (burgers.py, solvers/burgers_*.py)
- `u = -2 nu (ln phi)_x`, `phi0 = exp(-(W - min W)/2nu)`, `W = int w0` (spectral antiderivative,
  Nyquist zero), heat step = the CP7 circuit with `c = 0` (no advection layer, so no
  advection gates in CP8 counts). `cole_hopf_phi0` raises on a nonzero mean and when the
  valley `exp(-range)` underflows (range > 600).
- Two independent references (`burgers.py`, no cudaq): `exact_cole_hopf` (closed form,
  `u = sum ((x-y)/t) w / sum w` with log-domain softmax weights over a fine grid and
  periodic images; `exact_log_phi` is the same sum for `ln phi`) and
  `pseudospectral_burgers` (integrating-factor RK4, 2/3 dealiasing, N = 1024). They agree
  to ~1e-13 on smooth cases and ~2e-11 at nu = 0.02, T = 1; test tolerance =
  3 x (integrator's N -> 2N difference) + 1e-12. `discrete_burgers` is what the circuit
  computes on 2**n points (FD velocity `-(nu/h)(ln phi_{j+1} - ln phi_{j-1})`).
- Zero mean: `galilean_split` (asserts `|mean| <= 8 eps max|u0|`), `u(x,t) = U0 + w(x - U0 t, t)`;
  probes shift to `x + U0 T` exactly (`lab_positions`), fields via
  `evolve_spectral(w, T, c=U0)` (`shift_back`). The state carries `phi`, so the shift acts
  on the READOUT.
- Kernels (`burgers_kernels.py`): `heat_{a,b2,c}_on` (block) and `_measured` (gate-loaded
  `phi0` via `oneshot.prepare_real`, `2**n - 2` CX; measure system then ancillas);
  `fourier=True` drops the final QFT for the spectrum. B1 is not used in CP8.
  `burgers_quantum.sample_heat` is hardware-path (shots only); bins are per grid index, or
  per bit-reversed register index with `fourier` (`fft_counts` reorders).
- Estimator (`burgers_estimator.py`): `u = -(nu/2h) ln(c_{j+1}/c_{j-1})` from raw accepted
  counts. Delta method: `Var = (nu/2h)^2 (1/(S q+) + 1/(S q-))`, `q = P_succ p`, `S` total
  shots; bias `(nu/4hS)(1/q+ - 1/q-)`. Exact multinomial Monte Carlo: variance within 5%
  from a smaller expected count of 30 (16% off at 10, wrong at 1), bias good from 30;
  hence `c_min = 30` (the plan said 10). Zero count -> plain estimate NaN (counted),
  Haldane `c + 1/2` always defined and unbiased to first order from ~3 counts.
- Dynamic range (`burgers_study.py`, `benchmarks/burgers_dynamic_range.py`): `p` spans
  `exp(2/nu)`; shots at a probe `~ (nu/2h)^2 (1/q+ + 1/q-)/eps^2` are set by the smaller
  `p`. `u(pi) = 0` for sin x (symmetry), so probes are pi/4, pi/2, 3pi/4. For sin x, T = 1,
  eps = 0.05, n = 6: best probe 2e4 to 5e5 shots, worst 1e6 (nu = 0.5), 1e9 (0.1), 2e14
  (0.05). At nu = 0.02 the grid solution is NOT the physical one (13 to 37 grid points off by
  more than 10% in `ln p` or non-positive, the heat kernel cannot damp mode N/2 by `e^-50`,
  the true valley is ~1e-43, below double precision): rows are flagged `resolved = False`.
  Sampled runs (5 seeds, up to 3e6 shots) at nu = 0.5, 0.1 match the prediction (max |z| 2.3).
- Noise floor (`burgers_study.NOISE_FLOOR = 1e-3`, parameter `noise_floor`): a probe with
  `min(p_{j+1}, p_{j-1})` below it is "unresolvable regardless of shots" (`resolvable_on_hardware`
  per probe). It is a MODELLING ASSUMPTION for combined gate and readout error per shot, to be
  replaced by device calibration data; the sampled confirmations are noiseless and ignore it.
  For sin x, n = 6: nu = 0.5 all three probes resolvable, nu = 0.1 and 0.05 only pi/4,
  nu = 0.02 none.
- Limits: readout shots `~ 1/(P_succ p_min) ~ exp(Delta/nu)` with `Delta = max W - min W`;
  `P_succ` stays ~0.7 (0.57 to 0.88 over all rows), so the heat step is not the bottleneck;
  loading `phi0` costs `2**n - 2` CX (exponential input cost).
- Spectrum of phi: Fourier-basis readout, error follows the multinomial prediction at every
  nu; `4 nu^2 sum k^2 P_k` is the `p`-weighted mean of `u^2`. No pointwise `u`.
- Resources (`burgers_resources.py`, `oneshot_resources.build_mirror(advection=False,
  loader=...)`): measured lowered CX equals the construction (`2**n - 2` loader +
  one-shot block). Realistic hardware configurations: route A at n <= 5 (102 CX incl. loader,
  6 qubits at n = 5); B2 fits 20/24 qubits only up to n = 5 (20 qubits, 168 CX); route C has
  `q = n` for every nu tried (2n qubits). Qubit count alone does not make a row realistic.

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
3. Burgers: Cole-Hopf to heat equation (the one-shot circuit with c = 0 evolves phi),
   done in CP8 with the dynamic-range cost quantified; next, a nonlinear variational
   ansatz that avoids the logarithmic readout.
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
- Never compute an angle or infidelity between nearly equal states as `1 - F` or
  `arccos/arcsin(sqrt(1 - F))`: the subtraction cancels (absolute error ~eps, so relative
  error eps/theta^2; the angle has an absolute floor sqrt(eps) ~ 1.5e-8). Use
  `cudaq_pde.metrics` (`state_angle(s)`, `infidelity/infidelities` = sin^2 of an atan2
  of the orthogonal residual, backward stable to a few eps absolute; sums via `np.sum`,
  not BLAS dots, for run-to-run determinism). float64 cannot do better than eps ABSOLUTE
  for generic stored vectors (relative eps only for exactly representable inputs).
  `variational.reference_errors` returns `(angle, infidelity, l2)`. Tolerances in tests
  that compare such quantities must come from this precision (a few eps), see
  `test_global_error_obeys_angle_accumulation_bound`.
- Writing style in docs/README: no em dashes.
