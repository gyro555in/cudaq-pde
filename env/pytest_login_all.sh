#!/bin/bash
# Run the whole CPU test suite on the login node in N shards, one pytest process each,
# so no process exceeds ~150 s of CPU (the login node kills at 300 s). Prints the CPU
# seconds of every shard. Extra arguments go to pytest.
# Usage (after `source env/activate_rosi.sh`): env/pytest_login_all.sh [N] [pytest args]
# Default N = 3 (about 70 s of CPU per shard at present; raise N when a shard nears 150 s).
N="${1:-3}"
[ $# -gt 0 ] && shift
status=0
for k in $(seq 1 "$N"); do
    echo "=== shard $k/$N ==="
    start=$(awk '{print $14+$15+$16+$17}' /proc/$$/stat)
    OMP_NUM_THREADS=1 taskset -c 0 pytest --run-slow --shard "$k/$N" "$@" || status=1
    end=$(awk '{print $14+$15+$16+$17}' /proc/$$/stat)
    echo "shard $k/$N CPU seconds (children): $(( (end - start) / $(getconf CLK_TCK) ))"
done
exit $status
