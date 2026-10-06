#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --time=15
#SBATCH --account=rz
#SBATCH --job-name=cudaq-gpu-check
#SBATCH --output=logs/%x_%j.out

# GPU verification script for CUDA-Q on ROSI A100

set -e

echo "=== Node Info ==="
hostname
nvidia-smi

echo ""
echo "=== Activate CUDA-Q Environment ==="
module load python/3.12.4
export PIP_CACHE_DIR=/bigdata/rz/ramakr74/.pip_cache
source /bigdata/rz/ramakr74/venvs/cudaq-pde/bin/activate

echo "Python: $(python --version)"
echo "CUDA-Q: $(python -c 'import cudaq; print(cudaq.__version__)')"

echo ""
echo "=== Run GPU Verification Test ==="
python /bigdata/rz/ramakr74/cudaq_pde/env/check_cudaq_gpu.py
