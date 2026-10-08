#!/bin/bash
#SBATCH --partition=cpu-genoa
#SBATCH --account=rz
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --job-name=cudaq-pde-shot-study
#SBATCH --output=logs/%x_%j.out

# CPU shot-noise study (qpp-cpu, fp64): n=3, 5 steps, shots 1e2..1e5, 10 seeds.
# Submit from the repo root:
#   sbatch benchmarks/shot_study.sbatch            # 1 thread (default)
#   sbatch --cpus-per-task=4 benchmarks/shot_study.sbatch 4
# The optional argument is the thread count for OpenMP and BLAS; it must not exceed
# --cpus-per-task. It is recorded in the result metadata.
set -eo pipefail
THREADS="${1:-1}"
cd "$SLURM_SUBMIT_DIR"
source env/activate_rosi.sh
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"

if [ "$THREADS" -gt "${SLURM_CPUS_PER_TASK:-1}" ]; then
    echo "THREADS=$THREADS exceeds --cpus-per-task=${SLURM_CPUS_PER_TASK:-1}" >&2
    exit 2
fi

# printed AFTER the exports, so this is the environment the python process sees
echo "=== environment after thread setup ==="
echo "THREADS=$THREADS"
env | grep -E '^(OMP_|OPENBLAS_|MKL_|SLURM_(JOB_ID|CPUS_PER_TASK|JOB_PARTITION|NODELIST))' | sort
echo "======================================"

python benchmarks/shot_study.py --n 3 --depth 2 --steps 5 \
    --shots 100 1000 10000 100000 --seeds 10 --threads "$THREADS" \
    --outdir "results/shot_study_${SLURM_JOB_ID}"
