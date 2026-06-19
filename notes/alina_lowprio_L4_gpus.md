# Using the lowprio L4 GPUs on ScienceCluster

Short guide for fitting cognitive / sequential-sampling models (bauer DDM/RDM,
HSSM, numpyro) on the UZH ScienceCluster GPUs.

## What the L4s are, and when to use them

ScienceCluster has several GPU types in the `lowprio` partition: small **L4**
(24 GB, 1 per node, usually several idle), and bigger **A100 / H100 / H200**
(8 per node, in heavy demand).

- **Use an L4 when:** your model fits in 24 GB and the bottleneck is a NUTS
  sampler over a moderate dataset — i.e. *almost every* bauer / HSSM behavioural
  fit. A hierarchical DDM/RDM over ~60 subjects × ~180 trials runs in ~1–3 h on
  one L4, and there are usually idle L4s so you **start immediately** instead of
  queueing behind everyone's A100 jobs.
- **Reach for an A100/H100 only when:** the model genuinely needs >24 GB, or
  you're running a very large batch where the ~2–3× speedup is worth a longer
  queue. For single behavioural fits it's not worth it.
- **`lowprio` = preemptible.** Jobs can be killed if a higher-priority job needs
  the node. For short L4 fits this almost never bites; for very long runs
  (>12 h) checkpoint or accept the small risk.

Rule of thumb: **L4 for behavioural model fitting, by default.**

## One-time setup

```bash
ssh sciencecluster
# GPU conda env with jax[cuda12] + numpyro + hssm (built once on a GPU node):
ls $HOME/data/conda/envs/bauer_cuda/bin/python   # should exist
```
The CUDA env must be built *on a GPU node* (so jax sees the GPU) — see bauer's
`scripts/slurm_jobs/build_cuda_env.sh`.

## Submitting a fit

Request **one L4** with `--gres=gpu:L4:1`. Minimal sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=myfit
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00          # QOS cap is between 24h and 36h; ask for what you need
#SBATCH --output=%x_%j.log

PY=$HOME/data/conda/envs/bauer_cuda/bin/python
export PYTHONUNBUFFERED=1
cd $HOME/git/<your_project>
$PY -u fit.py <args> --backend numpyro
```

Submit:
```bash
sbatch myfit.sh
# override resources without editing the file:
sbatch --gres=gpu:L4:1 --time=12:00:00 --job-name=run2 myfit.sh <extra args>
```

Key points that trip people up:
- **`--backend numpyro`** (JAX NUTS). On a single GPU also use
  `chain_method='vectorized'` so all chains run together on the one GPU — bauer
  sets this by default; in raw numpyro pass it explicitly.
- **Don't `module load cuda`** — the conda env (`jax[cuda12]`) ships its own
  CUDA runtime. Loading a system CUDA on top causes version clashes.
- **Walltime**: the partition is "infinite" but the QOS caps a single job
  (≈24 h works, 36 h is rejected with `Invalid qos specification`). Ask for a
  bit more than you expect.
- **`#SBATCH --output=%x_%j.log`** names the log `<jobname>_<jobid>.log` so you
  can find it.

## Monitoring

```bash
squeue -u $USER -o "%.10i %.18j %.8T %.10M %R"   # your jobs
sinfo -o "%P %G %t" | grep -i l4                 # idle L4s right now?
tail -f myfit_<jobid>.log                         # live progress (numpyro bar)
sacct -j <jobid> -o State,Elapsed,MaxRSS,ExitCode # after it finishes
```
If a job sits `PENDING` with reason `(Priority)` / `(Resources)`, an L4 will free
up soon; `(QOSMaxWallDurationPerJobLimit)` means you asked for too long.

## Example / template

Working template used in this project (generic bauer DDM/RDM runner, logs +
runtime tracking): `dyscalculic_ddm/run_fit.sh`. The bauer library and its
`scripts/slurm_jobs/` templates are on GitHub:

  https://github.com/ruffgroup/bauer  →  `bauer/scripts/slurm_jobs/run_fit.sh`

Copy `run_fit.sh`, point `ENV_NAME`/`REPO` at yours, and submit with
`sbatch --gres=gpu:L4:1 run_fit.sh <model> <args>`.

## Gotchas specific to sequential-sampling fits

- `target_accept=0.99` is safe but slow (deep NUTS trees); `0.95` is ~2× faster
  and usually fine — but check r̂/ESS afterwards.
- Hierarchical DDM/RDM can hit hard geometry (e.g. `a`↔`t0` funnels). If r̂ is
  high, try more `tune` first; if it's *tune-invariant*, it's a structural
  ridge — reparameterize rather than burn more compute.
- Always check convergence before interpreting: `az.summary(idata)` → max r̂ ≤
  1.01, ESS in the hundreds+, few divergences.
