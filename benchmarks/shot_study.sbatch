#!/bin/bash
#SBATCH --partition=cpu-genoa
#SBATCH --account=rz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --job-name=cudaq-pde-shot-study
#SBATCH --output=logs/%x_%j.out

# CPU shot-noise study (qpp-cpu, fp64): n=3, 5 steps, shots 1e2..1e5, 10 seeds.
# Estimated 25 to 30 minutes of wall time. Submit from the repo root:
#   sbatch benchmarks/shot_study.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source env/activate_rosi.sh
export OMP_NUM_THREADS=1

python benchmarks/shot_study.py --n 3 --depth 2 --steps 5 \
    --shots 100 1000 10000 100000 --seeds 10 \
    --outdir "results/shot_study_${SLURM_JOB_ID}"
