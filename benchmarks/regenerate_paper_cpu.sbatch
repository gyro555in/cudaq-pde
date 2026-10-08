#!/bin/bash
#SBATCH --partition=cpu-genoa
#SBATCH --account=rz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --job-name=cudaq-pde-regenerate
#SBATCH --output=logs/%x_%j.out

# Every CPU-only paper output from one clean commit (all steps of
# benchmarks/regenerate_paper.sh, including the batch tier the login node skips), then
# the provenance check. One CPU, as in shot_study.sbatch; the steps keep their
# `taskset -c 0` pin. The time limit is a guess: no benchmark script has a recorded run
# time in logs/ (see the header of regenerate_paper.sh). Submit from the repo root, from
# a clean checkout at origin/main, only with approval:
#   sbatch benchmarks/regenerate_paper_cpu.sbatch
#   GPU_COMMIT=<sha of a gpu_release.sbatch run> sbatch --export=ALL benchmarks/regenerate_paper_cpu.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source env/activate_rosi.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

benchmarks/regenerate_paper.sh
