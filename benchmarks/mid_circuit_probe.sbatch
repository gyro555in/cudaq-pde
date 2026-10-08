#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=rz
#SBATCH --gpus=1
#SBATCH --exclude=ga008
#SBATCH --time=15
#SBATCH --job-name=cudaq-pde-mcm-probe
#SBATCH --output=logs/%x_%j.out

# Mid-circuit mz + reset probe and one-shot routes on one A100 (nvidia, fp64).
# Submit from the repo root:
#   sbatch benchmarks/mid_circuit_probe.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source env/activate_rosi.sh

nvidia-smi

python benchmarks/mid_circuit_probe.py --target nvidia --shots 20000

# fp64 state-vector agreement of routes A, B2, C (prints the log lines, writes
# benchmarks/output/oneshot_fp64_agreement.json with run metadata)
python benchmarks/oneshot_fp64_agreement.py --target nvidia --option fp64
