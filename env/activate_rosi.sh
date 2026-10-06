#!/bin/bash
# Activate CUDA-Q environment on ROSI (HZDR)
# Usage: source env/activate_rosi.sh

# Load Python module (required for shared libraries)
module load python/3.12.4

# Set pip cache
export PIP_CACHE_DIR=/bigdata/rz/ramakr74/.pip_cache

# Activate venv (located outside repo)
source /bigdata/rz/ramakr74/venvs/cudaq-pde/bin/activate

echo "CUDA-Q environment activated (ROSI)"
echo "  Python: $(python --version 2>&1)"
echo "  Venv: $VIRTUAL_ENV"
echo "  PIP_CACHE_DIR: $PIP_CACHE_DIR"
