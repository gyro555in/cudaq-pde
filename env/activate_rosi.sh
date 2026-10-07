#!/bin/bash
# Activate CUDA-Q environment on ROSI (HZDR)
# Usage: source env/activate_rosi.sh

# Load Python module (required for shared libraries)
module load python/3.12.4

# Set pip cache
export PIP_CACHE_DIR=/bigdata/rz/ramakr74/.pip_cache

# Login node: the qpp-cpu simulator spawns OpenMP threads on every core and
# burns the 300 s CPU cap. Inside a Slurm job (SLURM_JOB_ID set) keep defaults.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    export OMP_NUM_THREADS=1
fi

# Activate venv (located outside repo)
source /bigdata/rz/ramakr74/venvs/cudaq-pde/bin/activate

echo "CUDA-Q environment activated (ROSI)"
echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS:-unset}"
echo "  Python: $(python --version 2>&1)"
echo "  Venv: $VIRTUAL_ENV"
echo "  PIP_CACHE_DIR: $PIP_CACHE_DIR"
