#!/bin/bash
# Run pytest on the login node, pinned to one core with one OpenMP thread.
# CUDA-Q runtime threads otherwise spin on every core and burn the 300 s CPU cap.
# Usage (after `source env/activate_rosi.sh`): env/pytest_login.sh [pytest args]
# Inside a Slurm job run plain pytest instead.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "env/pytest_login.sh is for the login node only; inside a Slurm job (SLURM_JOB_ID=$SLURM_JOB_ID) run plain pytest." >&2
    exit 1
fi
OMP_NUM_THREADS=1 exec taskset -c 0 pytest "$@"
