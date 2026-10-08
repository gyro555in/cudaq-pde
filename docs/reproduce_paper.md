# Reproducing the paper's tables and figures

Every number and figure in the paper comes from a stored JSON file written by one script
of this repository. This page maps each of them to the exact command and the output path;
`tests/test_reproduce_paper.py` checks that every script named here exists, that every
output file name is one the named script writes, and that every benchmark script that
records run metadata appears in the table.

## Conventions

- Run from the repository root after `source env/activate_rosi.sh`.
- Login node: keep the `taskset -c 0` prefix (one core, CPU time equals wall time; a
  process is killed after 300 s of CPU). Inside a Slurm job drop the prefix.
- `sbatch` commands are submitted from the repository root and go to a compute node;
  GPU jobs (`mid_circuit_probe.sbatch`) are submitted only with approval.
- `<jobid>` is the Slurm job id. Outputs under `benchmarks/output/`, `results/` and
  `logs/` are gitignored; each JSON carries `collect_metadata()` (CUDA-Q version, target,
  GPU, driver, commit, `git_dirty`). Run from a clean commit, otherwise `git_dirty` is
  true and the recorded commit is not exactly the code that produced the file.

## Numbering

The numbers are the paper's:

| Paper item | Source |
|---|---|
| Table 1 | spectral resources |
| Table 2 | cost circuits (LCU against Hadamard test) and shift / LCU block resources |
| Table 3 | variational per-step table |
| Table 4 and Figure 2 | shot study |
| Table 5 | one-shot resources |
| Table 6 | Burgers resources (device fit) |
| Table 7 | mid-circuit probe: GPU job, emulated targets, record statistics |
| Table 8 and Figure 3 | Burgers dynamic range |
| Figure 4 | power spectrum shots |

A table made of several scripts has one row per script, all labelled with the same
number. Scripts that no paper item uses (`benchmarks/qiskit_roundtrip.py`, the two
examples) are not in the map; the test lists the benchmark among its exemptions.

## Data

| Paper item | Command | Script(s) | Output |
|---|---|---|---|
| Table 1 | `taskset -c 0 python benchmarks/spectral_resources.py` | `benchmarks/spectral_resources.py` | `benchmarks/output/spectral_resources.json` |
| Table 2 (cost circuits) | `taskset -c 0 python benchmarks/cost_circuit_comparison.py` | `benchmarks/cost_circuit_comparison.py` | `benchmarks/output/cost_circuit_comparison.json` |
| Table 2 (shift and LCU block resources) | `taskset -c 0 python benchmarks/measurable_resources.py` | `benchmarks/measurable_resources.py` | `benchmarks/output/measurable_resources.json` |
| Table 3 (and the depth and tolerance sweeps) | `taskset -c 0 python benchmarks/variational_report.py --ns 3 4` | `benchmarks/variational_report.py` | `benchmarks/output/variational_report.json` |
| Table 4 and Figure 2 data (32 min, 1 CPU, job 606305) | `sbatch benchmarks/shot_study.sbatch` | `benchmarks/shot_study.sbatch`, `benchmarks/shot_study.py` | `results/shot_study_<jobid>/shot_study.json`, `logs/cudaq-pde-shot-study_<jobid>.out` |
| Table 5 | `taskset -c 0 python benchmarks/oneshot_resources.py` | `benchmarks/oneshot_resources.py` | `benchmarks/output/oneshot_resources.json` |
| Table 6 | `taskset -c 0 python benchmarks/burgers_resources.py` | `benchmarks/burgers_resources.py` | `benchmarks/output/burgers_resources.json` |
| Table 7 (GPU probe and fp64 agreement of routes A, B2, C; needs approval) | `sbatch benchmarks/mid_circuit_probe.sbatch` | `benchmarks/mid_circuit_probe.sbatch`, `benchmarks/mid_circuit_probe.py`, `benchmarks/oneshot_fp64_agreement.py` | `benchmarks/output/mid_circuit_<target>.json` (target = nvidia), `benchmarks/output/oneshot_fp64_agreement.json`, `logs/cudaq-pde-mcm-probe_<jobid>.out` |
| Table 7 (CPU simulator) | `taskset -c 0 python benchmarks/mid_circuit_probe.py --target qpp-cpu` | `benchmarks/mid_circuit_probe.py` | `benchmarks/output/mid_circuit_<target>.json` |
| Table 7 (IonQ, emulated) | `taskset -c 0 python benchmarks/mid_circuit_probe.py --target ionq --emulate` | `benchmarks/mid_circuit_probe.py` | `benchmarks/output/mid_circuit_<target>_emulated.json` |
| Table 7 (Quantinuum, emulated) | `taskset -c 0 python benchmarks/mid_circuit_probe.py --target quantinuum --emulate` | `benchmarks/mid_circuit_probe.py` | `benchmarks/output/mid_circuit_<target>_emulated.json` |
| Table 7 (IQM, emulated) | `taskset -c 0 python benchmarks/mid_circuit_probe.py --target iqm --emulate` | `benchmarks/mid_circuit_probe.py` | `benchmarks/output/mid_circuit_<target>_emulated.json` |
| Table 7 (mid-circuit records, `cudaq.run` against `explicit_measurements`) | `taskset -c 0 python benchmarks/mid_circuit_probe_stats.py` | `benchmarks/mid_circuit_probe_stats.py` | `benchmarks/output/mid_circuit_probe_stats.json` |
| Table 8 and Figure 3 data | `taskset -c 0 python benchmarks/burgers_dynamic_range.py` | `benchmarks/burgers_dynamic_range.py` | `benchmarks/output/burgers_dynamic_range.json` |
| Figure 4 data | `taskset -c 0 python benchmarks/spectrum_shots.py` | `benchmarks/spectrum_shots.py` | `benchmarks/output/spectrum_shots.json` |

## Figures (read only the stored JSON above)

`taskset -c 0 python benchmarks/plots.py` draws every figure whose input exists and
prints the paths; `--outdir`, `--shot-study`, `--dynamic-range` and `--spectrum` select
other locations. Figure 2 uses the latest `results/shot_study_<jobid>/` directory.

| Paper item | Command | Script(s) | Output |
|---|---|---|---|
| Figure 2 (shot noise against shots per circuit) | `taskset -c 0 python benchmarks/plots.py` | `benchmarks/plots.py` | `results/figures/figure2_shot_study.png`, `results/figures/figure2_shot_study.pdf` |
| Figure 3 (Burgers cost against 1/nu, n = 6, sin, T = 1) | `taskset -c 0 python benchmarks/plots.py` | `benchmarks/plots.py` | `results/figures/figure3_dynamic_range.png`, `results/figures/figure3_dynamic_range.pdf` |
| Figure 4 (power spectrum error against shots) | `taskset -c 0 python benchmarks/plots.py` | `benchmarks/plots.py` | `results/figures/figure4_power_spectrum.png`, `results/figures/figure4_power_spectrum.pdf` |

## Regenerating everything from one commit

`benchmarks/regenerate_paper.sh` regenerates every stored CPU-only output (Tables 1, 2, 3,
5, 6, 8, the Figure 4 data, the CPU and emulated rows of Table 7, the record statistics,
`qiskit_roundtrip`, and the figures) and then runs the provenance checker. It refuses to
run unless the working tree is clean (untracked files count, because the run metadata
records them as `git_dirty`) and `HEAD` equals `origin/main`; it prints the commit.

```bash
benchmarks/regenerate_paper.sh                  # login node: only the steps marked login
sbatch benchmarks/regenerate_paper_cpu.sbatch   # compute node: every step (needs approval)
```

Run times. `logs/` records run times only for the test suite and the shot study, none for
a benchmark script. The times below were measured while developing the scripts, on the
pinned login node, and were not stored. A step is on the login path only if its time is
well below 150 s (the node kills at 300 s CPU); every unmeasured step is on the sbatch
path, and the login run skips it and says so.

| Step | Path | Time |
|---|---|---|
| `spectral_resources.py` | sbatch | unmeasured |
| `cost_circuit_comparison.py` | sbatch | unmeasured |
| `measurable_resources.py` | sbatch | unmeasured |
| `variational_report.py --ns 3 4` | sbatch | unmeasured |
| `oneshot_resources.py` | login | about 19 s |
| `burgers_resources.py` | login | about 2 s |
| `mid_circuit_probe.py --target qpp-cpu` | sbatch | unmeasured |
| `mid_circuit_probe.py --target ionq`, `quantinuum`, `iqm` with `--emulate` | sbatch | unmeasured |
| `mid_circuit_probe_stats.py` | login | about 15 s |
| `burgers_dynamic_range.py` | login | about 7 s (28 s wall) |
| `spectrum_shots.py` | sbatch | unmeasured (up to 1e6 shots, 20 seeds) |
| `qiskit_roundtrip.py` | login | 10 to 16 s |
| `plots.py` | sbatch | unmeasured |

The emulated runs exit with status 1 by design (`cudaq.run` is unsupported on emulated
IonQ and Quantinuum, IQM has no server); the script accepts 0 or 1 for them and requires
the JSON to be rewritten.

## Provenance check

```bash
python benchmarks/check_provenance.py
python benchmarks/check_provenance.py --commit <sha> --allow-commit-mismatch
python benchmarks/check_provenance.py --expect 'results/shot_study_606305/*=a562d15'
```

It prints file, commit, `git_dirty`, target, GPU and job id for every JSON in
`benchmarks/output/`, `results/shot_study_<jobid>/`, `results/gpu_release_<jobid>/` and
`logs/*_metadata.json`, and for every output this page names. Exit status 1 if a file is
missing, has `git_dirty` true, or carries a commit other than `HEAD`. Files that ran on a
GPU are checked against `--commit` when given; `--expect GLOB=SHA` sets the commit for
matching files; `--allow-commit-mismatch` downgrades commit mismatches to warnings (a
dirty file still fails). Most outputs stored before this page existed have `git_dirty`
true, so the check fails until they are regenerated.

## GPU release job

`benchmarks/gpu_release.sbatch` runs from the clean tree at `origin/main`: the GPU pytest
set (as `gpu_tests.sbatch`), the mid-circuit probe and `oneshot_fp64_agreement.py` (as
`mid_circuit_probe.sbatch`), and writes `logs/gpu_release_<jobid>_metadata.json`. It copies
its JSON files to `results/gpu_release_<jobid>/`, because the two probe outputs in
`benchmarks/output/` are overwritten by every job. It takes about 3 minutes (53 s and
109 s in the earlier jobs).

Card selection. ROSI has no node feature or typed gres for the memory of an A100, so the
card is chosen by node name. Measured with `nvidia-smi`: `ga007` is a 40 GB card (jobs
601171, 601192, 607242), `ga012` an 80 GB card (job 603240). `GPU_NODE` is the node for
`--nodelist`; it is set at submit time, not in the script, because the allocation is fixed
then. `EXPECT_GPU` (`40GB` or `80GB`) makes the job stop before running anything if
`nvidia-smi` reports another card. The two submit commands (not submitted; they need
approval):

```bash
GPU_NODE=ga007 EXPECT_GPU=40GB; sbatch --nodelist=$GPU_NODE --export=ALL,EXPECT_GPU=$EXPECT_GPU benchmarks/gpu_release.sbatch
GPU_NODE=ga012 EXPECT_GPU=80GB; sbatch --nodelist=$GPU_NODE --export=ALL,EXPECT_GPU=$EXPECT_GPU benchmarks/gpu_release.sbatch
```

The other 80 GB nodes (`ga010` to `ga015`, four GPUs each) are probably also 80 GB but
only `ga012` has been checked, and `ga008` is excluded (faulty). Afterwards run
`GPU_COMMIT=<sha> benchmarks/regenerate_paper.sh` or
`python benchmarks/check_provenance.py --commit <sha>`, where `<sha>` is the release commit
the GPU jobs ran on.

## Shot study: a record at its own commit

`results/shot_study_606305` (job 606305, 32 min, clean tree) was run at commit `a562d15`
and stays as the record; it is not rerun. Its commit differs from the release commit by
design. Files the shot study imports (`benchmarks/shot_study.py` and, transitively,
`classical`, `encoding`, `metadata`, `metrics`, `solvers/ansatz_numpy`, `solvers/lcu`,
`solvers/lcu_kernels`, `solvers/measurable`, `solvers/measurable_kernels`,
`solvers/sampling`, `solvers/spectral`, `solvers/spectral_kernels`, `solvers/variational`,
`solvers/variational_kernels`) that changed since `a562d15`, from
`git diff --stat a562d15..HEAD` over those files:

```
 benchmarks/shot_study.py              | 13 +++++--
 src/cudaq_pde/metadata.py             |  9 +++--
 src/cudaq_pde/metrics.py              | 65 +++++++++++++++++++++++++++++++++++
 src/cudaq_pde/solvers/ansatz_numpy.py | 10 ++++--
 src/cudaq_pde/solvers/measurable.py   |  8 +++--
 src/cudaq_pde/solvers/sampling.py     | 16 +++++++++
 src/cudaq_pde/solvers/variational.py  | 32 ++++++++++-------
 7 files changed, 130 insertions(+), 23 deletions(-)
```

What changed in each:

- `metrics.py` is new: stable angle and infidelity (`sin^2` of an `atan2`) instead of
  `1 - F`.
- `ansatz_numpy.py`, `variational.py`, `shot_study.py`: the cost and the reported
  infidelity use that form. The values agree with the old ones to a few `eps` absolute;
  the old `1 - F` could not resolve below about `1e-16`.
- `measurable.py`: `1 - n_zero / n_succ` became `(n_succ - n_zero) / n_succ` (exact
  integer difference), plus a docstring correction (heavy ball, not Adam). Same value up
  to rounding of the last bit.
- `sampling.py`: `run_records` added (a new function, not called by the shot study).
- `metadata.py`: four thread fields added to the metadata block.

The other imported modules are unchanged since `a562d15`. None of these changes alters
the algorithm, the circuits or the seeding; they can change results only at rounding level,
which a heavy-ball trajectory on sampled gradients can amplify, so a rerun is needed only
if bit-identical numbers are wanted.
