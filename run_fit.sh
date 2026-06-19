#!/bin/bash
#SBATCH --job-name=dyscalc_ddm
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null

# Fit a magnitude-comparison DDM on a GPU node.
# First positional arg = model label (ddm0 | ddm_group); default ddm0.
# Submit (GPU L4):
#   sbatch --gres=gpu:L4:1 --job-name=dyscalc_ddm0  run_fit.sh ddm0
#   sbatch --gres=gpu:L4:1 --job-name=dyscalc_group run_fit.sh ddm_group
# Extra args after the model override defaults, e.g. a quick test:
#   sbatch run_fit.sh ddm_group --n-subjects 6 --tune 1000
#
# Logs land at ~/logs/<jobname>_<jobid>.txt.

set -e

ENV_NAME="bauer_cuda"
REPO="$HOME/git/dyscalculic_ddm"
CSV="${RUN_CSV:-$REPO/data/magjudge_behavior_DNumRisk.csv}"   # override with RUN_CSV env (e.g. synthetic data)
OUT_DIR="/shares/zne.uzh/gdehol/dyscalculic_ddm"

MODEL="${1:-ddm0}"; [ "$#" -gt 0 ] && shift
OUT="$OUT_DIR/model-${MODEL}_trace.nc"
# Production defaults first; any extra CLI args ($@) append and override
# (argparse takes the last occurrence of a repeated option).
ARGS=("$CSV" --model "$MODEL" --out "$OUT" --backend numpyro \
      --n-subjects all --tune 2000 --draws 1000 --target-accept 0.99 "$@")

JOBNAME="${SLURM_JOB_NAME:-dyscalc_ddm}"
LOGFILE="$HOME/logs/${JOBNAME}_${SLURM_JOB_ID}.txt"
mkdir -p "$(dirname "$LOGFILE")" "$OUT_DIR"
exec >"$LOGFILE" 2>&1

T_START=$(date +%s)
echo "=============================================================="
echo "  job: $JOBNAME (id=$SLURM_JOB_ID) on $(hostname)"
echo "  start: $(date)   env: $ENV_NAME"
echo "  cmd: python -u fit_ddm.py ${ARGS[*]}"
echo "=============================================================="

# GPU envs ship their own CUDA runtime (jax[cuda12]); no `module load`.
# On a CPU-only node, pin JAX to CPU to silence the cuInit warning.
if [ -n "${SLURM_JOB_GPUS:-}" ] || [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi 2>/dev/null | head -12 || true
else
  export JAX_PLATFORMS=cpu
fi

export PYTHONUNBUFFERED=1
PY="$HOME/data/conda/envs/$ENV_NAME/bin/python"

cd "$REPO"
$PY -u fit_ddm.py "${ARGS[@]}"
rc=$?

T_END=$(date +%s); ELAPSED=$((T_END - T_START))
printf "  end: %s  exit=%d  elapsed=%ds\n" "$(date)" "$rc" "$ELAPSED"
exit $rc
