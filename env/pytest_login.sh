#!/bin/bash
# Run pytest on the login node, pinned to one core with one OpenMP thread.
# CUDA-Q runtime threads otherwise spin on every core and burn the 300 s CPU cap.
# Usage (after `source env/activate_rosi.sh`): env/pytest_login.sh [pytest args]
# Inside a Slurm job run plain pytest instead.
#
# One invocation must stay under about 150 s of CPU. The default run (slow tests
# skipped) is about 50 s. Running the slow tests needs shards, which this script
# insists on; env/pytest_login_all.sh runs every shard in turn.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "env/pytest_login.sh is for the login node only; inside a Slurm job (SLURM_JOB_ID=$SLURM_JOB_ID) run plain pytest." >&2
    exit 1
fi
case " $* " in
    *" --run-slow "*)
        case " $* " in
            *" --shard "*|*" --shard="*) ;;
            *)
                echo "--run-slow needs --shard K/N on the login node (one invocation must stay under ~150 s CPU); use env/pytest_login_all.sh [N]." >&2
                exit 2
                ;;
        esac
        ;;
esac
OMP_NUM_THREADS=1 exec taskset -c 0 pytest "$@"
