#!/bin/bash
# Regenerate every CPU-only stored paper output from one clean commit, then prove it
# with benchmarks/check_provenance.py. Run from the repository root after
# `source env/activate_rosi.sh`:
#
#   benchmarks/regenerate_paper.sh          # login node: only the steps marked login
#   sbatch benchmarks/regenerate_paper_cpu.sbatch   # compute node: every step
#
# Refuses to run unless the working tree is clean (`git status --porcelain` empty,
# untracked files included, because collect_metadata() counts them as git_dirty; move
# docs/paper_audit.md aside or use a fresh clone or worktree) and HEAD equals
# origin/main. Prints the commit.
#
# Which steps need the sbatch path. The only run times recorded in logs/ are for the
# test suite and the shot study; no benchmark script has a recorded time there. The times
# below are wall seconds I measured while developing each script on the pinned login
# node (taskset -c 0, so CPU time equals wall time) and did not store. A step is marked
# `login` only if that time is well below 150 s (the node kills at 300 s CPU). Every
# step with no measurement is marked `batch` and is skipped on the login node:
#
#   step                          tier   time
#   spectral_resources            batch  unmeasured (n up to 8, lowers circuits)
#   cost_circuit_comparison       batch  unmeasured (Monte Carlo over estimators)
#   measurable_resources          batch  unmeasured (lowering sweeps)
#   variational_report --ns 3 4   batch  unmeasured (L-BFGS, ~1e3 evaluations per step)
#   oneshot_resources             login  ~19 s
#   burgers_resources             login  ~2 s CPU
#   mid_circuit_probe qpp-cpu     batch  unmeasured (4000 shots x several checks)
#   mid_circuit_probe emulated    batch  unmeasured (x3: ionq, quantinuum, iqm)
#   mid_circuit_probe_stats       login  ~15 s CPU
#   burgers_dynamic_range         login  ~7 s CPU (28 s wall)
#   spectrum_shots                batch  unmeasured (up to 1e6 shots x 20 seeds, n 3..8)
#   qiskit_roundtrip              login  ~10 to 16 s
#   plots                         batch  unmeasured (figures from the stored JSON)
#
# On the login node the batch steps are skipped and the final check then fails until
# the sbatch run has refreshed them: that is the point of the check.
#
# The emulated mid_circuit_probe runs exit 1 by design (cudaq.run is not supported on
# emulated ionq and quantinuum, iqm has no server); the JSON is still written, so exit
# status 0 or 1 is accepted for them and the output file must be newer than the step.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is not clean; refusing to run:" >&2
    git status --short >&2
    exit 2
fi
git fetch --quiet origin main 2>/dev/null || echo "warning: git fetch failed, comparing with the local origin/main" >&2
HEAD_SHA=$(git rev-parse HEAD)
MAIN_SHA=$(git rev-parse origin/main)
if [ "$HEAD_SHA" != "$MAIN_SHA" ]; then
    echo "HEAD ($HEAD_SHA) is not origin/main ($MAIN_SHA); refusing to run" >&2
    exit 2
fi
echo "commit: $HEAD_SHA"

IN_SLURM=${SLURM_JOB_ID:+yes}
OUT=benchmarks/output
FAILED=0
SKIPPED=0
STAMP=$(mktemp)
trap 'rm -f "$STAMP"' EXIT

# step TIER TOLERATED_STATUS OUTPUT COMMAND...
# TOLERATED_STATUS: "0" or "0,1" (exit statuses that count as success)
step() {
    local tier=$1 ok=$2 out=$3
    shift 3
    if [ "$tier" = batch ] && [ -z "$IN_SLURM" ]; then
        echo "=== SKIP (batch tier, use sbatch): $*"
        SKIPPED=$((SKIPPED + 1))
        return
    fi
    echo "=== $*"
    touch "$STAMP"
    sleep 1
    local t0=$SECONDS
    "$@"
    local rc=$?
    echo "--- exit $rc after $((SECONDS - t0)) s"
    case ",$ok," in
        *",$rc,"*) ;;
        *) echo "FAILED: $*" >&2; FAILED=$((FAILED + 1)); return ;;
    esac
    if [ ! "$out" -nt "$STAMP" ]; then
        echo "FAILED: $out was not written by: $*" >&2
        FAILED=$((FAILED + 1))
    fi
}

PIN="taskset -c 0"
step batch 0 $OUT/spectral_resources.json $PIN python benchmarks/spectral_resources.py
step batch 0 $OUT/cost_circuit_comparison.json $PIN python benchmarks/cost_circuit_comparison.py
step batch 0 $OUT/measurable_resources.json $PIN python benchmarks/measurable_resources.py
step batch 0 $OUT/variational_report.json $PIN python benchmarks/variational_report.py --ns 3 4
step login 0 $OUT/oneshot_resources.json $PIN python benchmarks/oneshot_resources.py
step login 0 $OUT/burgers_resources.json $PIN python benchmarks/burgers_resources.py
step batch 0 $OUT/mid_circuit_qpp-cpu.json $PIN python benchmarks/mid_circuit_probe.py --target qpp-cpu
for t in ionq quantinuum iqm; do
    step batch 0,1 $OUT/mid_circuit_${t}_emulated.json $PIN python benchmarks/mid_circuit_probe.py --target "$t" --emulate
done
step login 0 $OUT/mid_circuit_probe_stats.json $PIN python benchmarks/mid_circuit_probe_stats.py
step login 0 $OUT/burgers_dynamic_range.json $PIN python benchmarks/burgers_dynamic_range.py
step batch 0 $OUT/spectrum_shots.json $PIN python benchmarks/spectrum_shots.py
step login 0 $OUT/qiskit_roundtrip.json $PIN python benchmarks/qiskit_roundtrip.py
step batch 0 results/figures/figure4_power_spectrum.pdf $PIN python benchmarks/plots.py

echo "=== steps failed: $FAILED, skipped (batch tier): $SKIPPED"
echo "=== provenance check"
# The shot study is a record at its own commit by design (docs/reproduce_paper.md).
# GPU outputs are checked against GPU_COMMIT (the commit a gpu_release.sbatch job ran
# on); unset, they are checked against HEAD and fail until that job has been rerun.
ARGS=(--expect "results/shot_study_606305/*=a562d15")
[ -n "${GPU_COMMIT:-}" ] && ARGS+=(--commit "$GPU_COMMIT")
python benchmarks/check_provenance.py "${ARGS[@]}"
CHECK=$?
if [ "$FAILED" -ne 0 ] || [ "$CHECK" -ne 0 ]; then
    echo "regeneration incomplete: $FAILED failed step(s), provenance check exit $CHECK" >&2
    exit 1
fi
echo "all stored outputs match $HEAD_SHA"
