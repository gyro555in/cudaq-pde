# CUDA-Q Environment Setup on ROSI (HZDR)

**Setup Date:** 2026-10-06
**Hostname:** rosi4 (login node), GPU nodes: ga006-ga015 (A100)

## System Information

| Item | Value |
|------|-------|
| **OS** | Ubuntu 22.04 LTS (Linux 5.15.0-118-generic) |
| **glibc** | 2.35 (manylinux_2_35 compatible) |
| **Architecture** | x86_64 |

## Python Environment

| Item | Value |
|------|-------|
| **Python Version** | 3.12.4 |
| **Python Module** | `python/3.12.4` (loaded via `module load`) |
| **venv Location** | `/bigdata/rz/ramakr74/venvs/cudaq-pde` (outside repo) |
| **Activation** | `source env/activate_rosi.sh` |
| **PIP_CACHE_DIR** | `/bigdata/rz/ramakr74/.pip_cache` |

## CUDA-Q Installation

| Item | Value |
|------|-------|
| **Package** | `cuda-quantum-cu12>=0.16,<0.17` |
| **Version Installed** | 0.16.0.post1 |
| **GPU Target** | `nvidia` with `option="fp64"` (double precision) |
| **CPU Target** | `qpp-cpu` (for testing on login node) |

### CUDA Runtime Strategy

**Important:** CUDA-Q wheels (`cuda-quantum-cu12`) bundle their own CUDA runtime libraries. Therefore:
- Do **NOT** load any CUDA module (`cuda/12.8`, etc.) in the environment.
- Only the GPU driver matters for target selection.
- The activation script (`activate_rosi.sh`) loads only `python/3.12.4`.

## Hardware

### Login Node (rosi4)
- CPU-only, used for code development and testing on qpp-cpu target

### GPU Nodes (ga006-ga015)
| Item | Value |
|------|-------|
| **GPU Model** | NVIDIA A100-SXM4-40GB |
| **Compute Capability** | 8.0 |
| **Memory per GPU** | 40 GB HBM2e |
| **Driver Version** | 570.158.01 |
| **Driver Support** | CUDA 12.x (cu12 is compatible) |
| **Partition** | `gpu-a100` |
| **Account** | `rz` (for sbatch priority) |

## Dependencies

### Core CUDA-Q & Simulation
- `cuda-quantum-cu12==0.16.0.post1` (includes cuStateVec, cuTensorNet, cuDensityMat)
- `numpy==2.5.3`
- `scipy==1.18.1`

### Development & Testing
- `pytest==9.1.1`
- `ruff==0.16.10` (linting & formatting)
- `matplotlib==3.11.2` (plotting for validation)

### Jupyter
- `ipykernel==7.4.0` (for interactive kernels)
- `jupyter-client==8.10.0`, `jupyter-core==5.9.1`

**Full lock file:** `env/requirements-lock.txt` (generated with `pip freeze`)

## Verification Results

### CPU Verification (Login Node)
Date: 2026-10-06
- **Test:** GHZ kernel (3 qubits) on `qpp-cpu`
- **Result:** Sampling (100 shots) and state vector extraction: **PASS**
- **State:** |000> and |111> with equal probability (normalized)

### GPU Verification (A100 Node)
Date: 2026-10-06
- **Job ID:** 599531 (node ga009)
- **Test 1 - GHZ Kernel (3 qubits):**
  - GPU target: `nvidia` (fp64)
  - Computation time: 0.1615s
  - CPU target: `qpp-cpu` computation time: 0.0003s
  - Max difference: 0.00e+00 (exact agreement to 1e-12)
  - **PASS**
- **Test 2 - 20-Qubit Circuit:**
  - Computation time: 0.0931s
  - State vector size: 2^20 = 1,048,576 complex elements
  - Norm: 1.0000000000 (verified)
  - **PASS**
- **Exit Code:** 0 (success)
- **Resources Used:** 15s CPU time, 40.32 kJ energy

## sbatch Template

For GPU jobs, use:
```bash
#SBATCH --partition=gpu-a100
#SBATCH --account=rz
#SBATCH --gpus=1
#SBATCH --time=15
#SBATCH --output=logs/%x_%j.out
```

Submit from the repo root.

Before submitting, activate the environment:
```bash
source env/activate_rosi.sh
```

## File Structure

```
cudaq_pde/
  env/
    activate_rosi.sh           (activation script)
    requirements-lock.txt      (pip freeze output)
    ENVIRONMENT.md             (this file)
    gpu_check.sbatch           (GPU verification job template)
  test_cudaq_cpu.py            (CPU target test)
  test_cudaq_gpu.py            (GPU target test)
  CLAUDE.md                     (project documentation)
  step2_gpu_check.sh           (system discovery script)
```

## Notes

- Never run GPU work on login node (rosi4).
- venv is tied to python/3.12.4; always activate via `env/activate_rosi.sh`.
- CUDA-Q 0.16 requires kernel definitions in `.py` files (not inline).
- GPU target uses fp64 precision per CLAUDE.md (cudaq.set_target("nvidia", option="fp64")).
- All reference benchmarks should include git commit hash, cudaq version, driver version, and GPU model in metadata.
