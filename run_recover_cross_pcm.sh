#!/bin/bash
#SBATCH --job-name=dys_rec_pcm
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null

# Cross-model recovery: simulate from the real rdm_max trace, refit the
# pure PCM (choice-only) to the simulated choices. One array task = one
# rep. CPU-only (PCM refit is a static probit, no accumulator likelihood).
# Submit:
#   sbatch --array=0-23 run_recover_cross_pcm.sh \
#       /shares/zne.uzh/gdehol/dyscalculic_ddm/model-rdm_max_cont_hn_trace.nc
set -e

ENV_NAME="bauer_cuda"
REPO="$HOME/git/dyscalculic_ddm"
TRACE="${1:?trace path}"
REP="${SLURM_ARRAY_TASK_ID:-0}"

JOBNAME="${SLURM_JOB_NAME:-dys_rec_pcm}"
LOGFILE="$HOME/logs/${JOBNAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${REP}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "=== $JOBNAME rep=$REP on $(hostname) $(date) ==="
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
# Same immutable v0.4.0 pin run_fit.sh uses -- do NOT drop this, the
# bauer_cuda env's pip-installed bauer can be a different (newer) checkout.
export PYTHONPATH="$HOME/bauer-030"
PY="$HOME/data/conda/envs/$ENV_NAME/bin/python"

cd "$REPO"
$PY -u recover_cross_pcm.py --trace "$TRACE" --rep "$REP" \
    --draws 800 --tune 800 --chains 4 "${@:2}"
echo "=== done rep=$REP $(date) ==="
