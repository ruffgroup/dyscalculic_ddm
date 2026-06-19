# dyscalculic_ddm

Computational modelling of a **symbolic magnitude-comparison** task (the
DNumRisk dataset: 66 subjects, 33 control / 33 dyscalculic, ~11.7k trials —
"see n1, then n2, choose the larger"). We fit a Bayesian-observer front-end
([bauer](https://github.com/ruffgroup/bauer)) crossed with three response
layers — static **choice**, **drift-diffusion (DDM)**, and **race-diffusion
(RDM)** — to ask *where* the dyscalculia deficit sits.

---

## ⚠️ Which bauer version — read this first

The models live in **bauer**, and the analyses depend on a specific version.
Use exactly this one:

> ### **bauer `v0.4.0`** — github.com/ruffgroup/bauer, tag `v0.4.0`

`environment.yml` already pins it, so `conda env create -f environment.yml`
just works. To install/upgrade by hand:

```bash
pip install --force-reinstall "git+https://github.com/ruffgroup/bauer.git@v0.4.0"
```

- ✅ The `v0.4.0` tag is **public and immutable** — it always resolves to the
  exact commit (`e406694`) the analyses were run with.
- ❌ Do **not** use `v0.3.0`, `main`, or any branch name. The 0.4.0 line lives
  on a branch that *diverged* from the release line, so anything else gives you
  subtly different model code. (Quirk: the in-tree `__version__` string on this
  commit still reads `0.3.0` — ignore it; the `v0.4.0` **tag** is authoritative.)

---

## The key figure: `notes/figures/overview_verdict.pdf`

This single multi-page PDF is **the** output everything hangs on — the
dyscalc−control contrast for every parameter, in two noise decompositions, with
posterior-predictive checks (psychometric, RT, QPP), per-group SDs, and LOO
model comparison. **It is committed in this repo**, so you have it the moment
you clone.

To **regenerate it locally** (no cluster, no 41 GB of traces needed — the small
group-level summary it's built from is committed):

```bash
python plot_overview.py        # reads notes/data/overview.{json,npz}
                               # writes notes/figures/overview_verdict.pdf
```

## What the figure shows

The model to trust is the **advantage race diffusion model (RDM)**; the DDM
can't reproduce the magnitude effect and loses on LOO by ~730 (p8–10). On the
RDM, dyscalculics show:

- **A noisier number representation** (P = 1.00): perceptual noise 0.34 → 0.45;
  **memory noise roughly doubles, 0.049 → 0.091** — the held first number is hit
  hardest.
- **A compensatory threshold increase** (`a` 0.50 → 0.65, P = 1.00): so in raw
  behaviour they're clearly slower but only *slightly* less accurate.
- **Higher urgency** (`w_0`), a **slightly larger magnitude effect** (`w_s`), and
  a **larger compression effect** (more regression to the prior).

One line: **noisier → slower, with stronger compression, especially in memory.**

For the full reasoning and numbers, read
[`notes/CONCLUSIONS.md`](notes/CONCLUSIONS.md) and
[`notes/MODELS.md`](notes/MODELS.md) — those notes are the source of truth and
move faster than this README.

## Where to start reading

| Read this | For |
|---|---|
| [`notes/BRIEF_FOR_MAIKE.md`](notes/BRIEF_FOR_MAIKE.md) | **Start here** — one-page orientation + how to get the key figure. |
| [`notes/CONCLUSIONS.md`](notes/CONCLUSIONS.md) | The verdict: what's robust, what's still open. |
| [`notes/MODELS.md`](notes/MODELS.md) | Model registry, the fitting recipe, the derived quantities. |
| `notes/figures/overview_verdict.pdf` | The summary figure all the conclusions hang on. |

## Repository layout

```
fit_ddm.py            build_model(spec, df) + fit; the spec mini-language.   [cluster GPU]
run_fit.sh            SLURM wrapper for fit_ddm.py.                          [cluster]
collect_overview.py   aggregate traces -> notes/data/overview.{json,npz}.    [cluster]
plot_overview.py      render notes/figures/overview_verdict.pdf.             [local]
recover_bootstrap.py  parameter-recovery harness (run on the cluster via     [cluster]
aggregate_recovery.py   run_recover.sh; commands in notes/CONCLUSIONS.md).
run_recover.sh
summarize_traces.py   trace diagnostics: convergence / contrasts / ridge.   [cluster]
group_contrasts.py
chain_diag.py
ridge_diag.py
figstyle.py           shared plotting style (imported by the plotters).
notes/                CONCLUSIONS, MODELS, the brief, and the two key figures.
data/        (gitignored) input behavioural CSV — see "Get the data".
results/     (gitignored) 41 GB of posterior traces — see "Get the traces".
```

Most scripts carry a usage block in their top docstring; read it before running.

## Setup

```bash
conda env create -f environment.yml     # installs bauer v0.4.0 (see top)
conda activate dyscalculic_ddm
```

On a cluster GPU node, install `jax[cuda12]` in the env instead of the CPU numpyro.

## Get the data

The behavioural CSV is human-subjects data and is **not** in this public repo.
Ask Gilles for `data/magjudge_behavior_DNumRisk.csv` (raw, 66 subjects — the
input everything reads) and drop it into `data/`.

## Get the traces (optional)

Only needed if you want to **refit or re-aggregate** rather than just re-plot
the committed figure. The 41 GB of posterior traces live on the cluster:

```bash
rsync -a sciencecluster:/shares/zne.uzh/gdehol/dyscalculic_ddm/ results/
```

## Reproduce, end to end

```bash
# 1. FIT (cluster GPU L4). Each model label maps to a bauer build in fit_ddm.py.
sbatch --gres=gpu:L4:1 --job-name=dyscalc_group run_fit.sh ddm_group
#   traces land in /shares/zne.uzh/gdehol/dyscalculic_ddm/model-<label>_trace.nc

# 2. AGGREGATE (cluster) — reduce the traces to one small summary.
python collect_overview.py        # -> notes/data/overview.{json,npz}

# 3. PLOT (local) — render the key figure from the summary.
python plot_overview.py           # -> notes/figures/overview_verdict.pdf
```

The split is deliberate: **fit + aggregate on the cluster** (large traces, GPU),
**plot + inspect locally** (small summaries). See `notes/MODELS.md §6` for the
per-script role table.
