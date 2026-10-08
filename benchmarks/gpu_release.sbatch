#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=rz
#SBATCH --gpus=1
#SBATCH --exclude=ga008
#SBATCH --time=20
#SBATCH --job-name=cudaq-pde-gpu-release
#SBATCH --output=logs/%x_%j.out

# All GPU evidence of the paper in one job, from a clean checkout of origin/main:
# the GPU pytest set (as gpu_tests.sbatch), the mid-circuit probe and the fp64
# agreement of routes A, B2, C (as mid_circuit_probe.sbatch), and the job's metadata.
# Previous runs took 53 s (tests, job 603240) and 109 s (probe, job 607242).
#
# Choosing the card. ROSI has no node feature or typed gres that tells a 40 GB from an
# 80 GB A100 (`--gpus=1` gets whichever node is free), so the card is chosen by node
# name. Verified with nvidia-smi: ga007 is 40 GB (jobs 601171, 601192, 607242), ga012 is
# 80 GB (job 603240). Two variables, set at submit time:
#   GPU_NODE    node for `--nodelist` (it cannot be set inside the script: the
#               allocation is fixed at submit time)
#   EXPECT_GPU  "40GB" or "80GB": the job stops before running anything if nvidia-smi
#               reports a different card, so a wrong node name cannot go unnoticed
# Exact commands, from the repo root, only with approval:
#   GPU_NODE=ga007 EXPECT_GPU=40GB; sbatch --nodelist=$GPU_NODE --export=ALL,EXPECT_GPU=$EXPECT_GPU benchmarks/gpu_release.sbatch
#   GPU_NODE=ga012 EXPECT_GPU=80GB; sbatch --nodelist=$GPU_NODE --export=ALL,EXPECT_GPU=$EXPECT_GPU benchmarks/gpu_release.sbatch
#
# Outputs. benchmarks/output/mid_circuit_nvidia.json and oneshot_fp64_agreement.json are
# overwritten by every job, so each job also copies its JSON files and the pytest log to
# results/gpu_release_<jobid>/ (one directory per card). The job id is in the metadata.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source env/activate_rosi.sh

# clean tree at origin/main (untracked files count: collect_metadata() sees them)
if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is not clean; refusing to run:" >&2
    git status --short >&2
    exit 2
fi
HEAD_SHA=$(git rev-parse HEAD)
if [ "$HEAD_SHA" != "$(git rev-parse origin/main)" ]; then
    echo "HEAD ($HEAD_SHA) is not origin/main; refusing to run" >&2
    exit 2
fi
echo "commit: $HEAD_SHA"

nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "GPU: $GPU_NAME on $(hostname -s) (EXPECT_GPU=${EXPECT_GPU:-unset})"
if [ -n "${EXPECT_GPU:-}" ] && [[ "$GPU_NAME" != *"$EXPECT_GPU"* ]]; then
    echo "expected a $EXPECT_GPU card, got '$GPU_NAME'; stopping" >&2
    exit 3
fi

OUTDIR="results/gpu_release_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

python - <<PY
import cudaq
from cudaq_pde.metadata import write_metadata
cudaq.set_target("nvidia", option="fp64")
meta = write_metadata("logs/gpu_release_${SLURM_JOB_ID}_metadata.json")
print("metadata:", meta)
PY

pytest --run-gpu --run-slow -m gpu -v 2>&1 | tee "$OUTDIR/pytest_gpu.log"

python benchmarks/mid_circuit_probe.py --target nvidia --shots 20000
python benchmarks/oneshot_fp64_agreement.py --target nvidia --option fp64

cp benchmarks/output/mid_circuit_nvidia.json benchmarks/output/oneshot_fp64_agreement.json "$OUTDIR/"
cp "logs/gpu_release_${SLURM_JOB_ID}_metadata.json" "$OUTDIR/"

# the rows of this job must show this commit, git_dirty False and this job id
# (other rows are filtered out; the full table is for regenerate_paper.sh)
python benchmarks/check_provenance.py --commit "$HEAD_SHA" \
    | grep -E "gpu_release_${SLURM_JOB_ID}|expected|files," || true
