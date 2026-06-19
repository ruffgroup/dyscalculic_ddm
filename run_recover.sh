#!/bin/bash
#SBATCH --job-name=dys_recover
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

# Parametric-bootstrap recovery, one array task = one rep.
#   MODEL=$1  TRACE=$2 (relative to repo)
# Submit:
#   choice (CPU array):
#     sbatch --array=0-99 --job-name=dys_rec_choice run_recover.sh choice \
#        /shares/zne.uzh/gdehol/dyscalculic_ddm/model-choice_group_trace.nc
#   ddm / ddm_sv (GPU array, throttle concurrency with %N):
#     sbatch --array=0-99%16 --gres=gpu:L4:1 --job-name=dys_rec_ddm \
#        run_recover.sh ddm  /shares/zne.uzh/gdehol/dyscalculic_ddm/model-ddm_group_trace.nc
#     sbatch --array=0-99%16 --gres=gpu:L4:1 --job-name=dys_rec_ddmsv \
#        run_recover.sh ddm_sv /shares/zne.uzh/gdehol/dyscalculic_ddm/model-ddm_sv_group_trace.nc
set -e

ENV_NAME="bauer_cuda"
REPO="$HOME/git/dyscalculic_ddm"
MODEL="${1:?model}"; TRACE="${2:?trace path}"
REP="${SLURM_ARRAY_TASK_ID:-0}"

JOBNAME="${SLURM_JOB_NAME:-dys_recover}"
LOGFILE="$HOME/logs/${JOBNAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${REP}.txt"
mkdir -p "$(dirname "$LOGFILE")"
exec >"$LOGFILE" 2>&1

echo "=== $JOBNAME model=$MODEL rep=$REP on $(hostname) $(date) ==="
if [ -n "${SLURM_JOB_GPUS:-}" ] || [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi 2>/dev/null | head -10 || true
else
  export JAX_PLATFORMS=cpu
fi
export PYTHONUNBUFFERED=1
PY="$HOME/data/conda/envs/$ENV_NAME/bin/python"

cd "$REPO"
# Defaults below; any extra args after MODEL TRACE ("${@:3}") override them
# (argparse takes the last occurrence), e.g. run_recover.sh ddm <trace> --tune 2000 --draws 1000
$PY -u recover_bootstrap.py --model "$MODEL" --trace "$TRACE" --rep "$REP" \
    --draws 800 --tune 800 --chains 4 "${@:3}"
echo "=== done rep=$REP $(date) ==="
