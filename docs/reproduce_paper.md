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

The repository does not carry the paper's numbering. The paper items below are the ones
the author named (Table 3 = variational per-step table, Table 7 = Burgers dynamic range,
Figures 2, 3, 4) and Table 1 (spectral resources, assigned by content in
`docs/paper_audit.md` item 25: confirm). Tables 2, 4, 5 and 6 are not identified: their
candidate sources are the rows marked "unassigned".

## Data

| Paper item | Command | Script(s) | Output |
|---|---|---|---|
| Table 1 | `taskset -c 0 python benchmarks/spectral_resources.py` | `benchmarks/spectral_resources.py` | `benchmarks/output/spectral_resources.json` |
| Table 3 (and the depth and tolerance sweeps) | `taskset -c 0 python benchmarks/variational_report.py --ns 3 4` | `benchmarks/variational_report.py` | `benchmarks/output/variational_report.json` |
| Table 7 and Figure 3 data | `taskset -c 0 python benchmarks/burgers_dynamic_range.py` | `benchmarks/burgers_dynamic_range.py` | `benchmarks/output/burgers_dynamic_range.json` |
| Figure 2 data (32 min, 1 CPU, job 606305) | `sbatch benchmarks/shot_study.sbatch` | `benchmarks/shot_study.sbatch`, `benchmarks/shot_study.py` | `results/shot_study_<jobid>/shot_study.json`, `logs/cudaq-pde-shot-study_<jobid>.out` |
| Figure 4 data | `taskset -c 0 python benchmarks/spectrum_shots.py` | `benchmarks/spectrum_shots.py` | `benchmarks/output/spectrum_shots.json` |
| unassigned: LCU against Hadamard-test cost circuits | `taskset -c 0 python benchmarks/cost_circuit_comparison.py` | `benchmarks/cost_circuit_comparison.py` | `benchmarks/output/cost_circuit_comparison.json` |
| unassigned: shift and LCU block resources | `taskset -c 0 python benchmarks/measurable_resources.py` | `benchmarks/measurable_resources.py` | `benchmarks/output/measurable_resources.json` |
| unassigned: one-shot resources, n = 3 to 10, sigma sweep | `taskset -c 0 python benchmarks/oneshot_resources.py` | `benchmarks/oneshot_resources.py` | `benchmarks/output/oneshot_resources.json` |
| unassigned: Burgers resources and device fit | `taskset -c 0 python benchmarks/burgers_resources.py` | `benchmarks/burgers_resources.py` | `benchmarks/output/burgers_resources.json` |
| unassigned: mid-circuit records, `cudaq.run` against `explicit_measurements` | `taskset -c 0 python benchmarks/mid_circuit_probe_stats.py` | `benchmarks/mid_circuit_probe_stats.py` | `benchmarks/output/mid_circuit_probe_stats.json` |
| unassigned: mid-circuit probe on one target (CPU or emulated) | `taskset -c 0 python benchmarks/mid_circuit_probe.py --target qpp-cpu` | `benchmarks/mid_circuit_probe.py` | `benchmarks/output/mid_circuit_<target>.json` |
| unassigned: GPU probe and fp64 agreement of routes A, B2, C (needs approval) | `sbatch benchmarks/mid_circuit_probe.sbatch` | `benchmarks/mid_circuit_probe.sbatch`, `benchmarks/mid_circuit_probe.py`, `benchmarks/oneshot_fp64_agreement.py` | `benchmarks/output/mid_circuit_<target>.json` (target = nvidia), `benchmarks/output/oneshot_fp64_agreement.json`, `logs/cudaq-pde-mcm-probe_<jobid>.out` |
| unassigned: fp64 agreement of the one-shot routes on the CPU | `taskset -c 0 python benchmarks/oneshot_fp64_agreement.py --target qpp-cpu` | `benchmarks/oneshot_fp64_agreement.py` | `benchmarks/output/oneshot_fp64_agreement.json` |
| unassigned: OpenQASM round trip through Qiskit (`pip install -e ".[aqt]"`) | `taskset -c 0 python benchmarks/qiskit_roundtrip.py` | `benchmarks/qiskit_roundtrip.py` | `benchmarks/output/qiskit_roundtrip.json`, `results/qasm_export/<circuit>_n<n>.qasm` |
| unassigned: spectral advection example | `taskset -c 0 python examples/advection_spectral.py --n 6` | `examples/advection_spectral.py` | `results/advection_spectral.json`, `results/advection_spectral.png` |
| unassigned: variational example | `taskset -c 0 python examples/advection_diffusion_variational.py --n 3` | `examples/advection_diffusion_variational.py` | `results/advection_diffusion_variational.json`, `results/advection_diffusion_variational.png` |

## Figures (read only the stored JSON above)

`taskset -c 0 python benchmarks/plots.py` draws every figure whose input exists and
prints the paths; `--outdir`, `--shot-study`, `--dynamic-range` and `--spectrum` select
other locations. Figure 2 uses the latest `results/shot_study_<jobid>/` directory.

| Paper item | Command | Script(s) | Output |
|---|---|---|---|
| Figure 2 (shot noise against shots per circuit) | `taskset -c 0 python benchmarks/plots.py` | `benchmarks/plots.py` | `results/figures/figure2_shot_study.png`, `results/figures/figure2_shot_study.pdf` |
| Figure 3 (Burgers cost against 1/nu, n = 6, sin, T = 1) | `taskset -c 0 python benchmarks/plots.py` | `benchmarks/plots.py` | `results/figures/figure3_dynamic_range.png`, `results/figures/figure3_dynamic_range.pdf` |
| Figure 4 (power spectrum error against shots) | `taskset -c 0 python benchmarks/plots.py` | `benchmarks/plots.py` | `results/figures/figure4_power_spectrum.png`, `results/figures/figure4_power_spectrum.pdf` |
