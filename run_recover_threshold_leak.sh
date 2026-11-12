#!/bin/bash
#SBATCH --job-name=dys_leak
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null

# Interventional cross-model recovery: rescale ONE group contrast in the real
# rdm_max posterior draw, simulate, refit the choice-only PCM. One array task =
# one (rep, condition) pair; 7 conditions, so task = rep*7 + cond_ix. CPU-only.
# Submit:
#   sbatch --array=0-279%120 run_recover_threshold_leak.sh \
#       /shares/zne.uzh/gdehol/dyscalculic_ddm/model-rdm_max_cont_hn_trace.nc
set -e

ENV_NAME="bauer_cuda"
REPO="$HOME/git/dyscalculic_ddm"
TRACE="${1:?trace path}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

JOBNAME="${SLURM_JOB_NAME:-dys_leak}"
LOGFILE="$HOME/logs/${JOBNAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "=== $JOBNAME task=$TASK on $(hostname) $(date) ==="
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
# Same immutable v0.4.0 pin run_fit.sh / run_recover_cross_pcm.sh use -- do NOT
# drop this, the bauer_cuda env's pip-installed bauer can be a different checkout.
export PYTHONPATH="$HOME/bauer-030"
PY="$HOME/data/conda/envs/$ENV_NAME/bin/python"

cd "$REPO"
$PY -u recover_threshold_leak.py --trace "$TRACE" --task "$TASK" \
    --draws 800 --tune 800 --chains 4 "${@:2}"
echo "=== done task=$TASK $(date) ==="
