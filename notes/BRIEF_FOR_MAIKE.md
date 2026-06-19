# Brief for Maike — dyscalculic_ddm

Hi Maike — this is the magnitude-comparison modelling project (DNumRisk: 66
subjects, 33 control / 33 dyscalculic, "see n1, then n2, choose the larger").
Everything is now on GitHub at **`github.com/ruffgroup/dyscalculic_ddm`** so you
can clone it and run the analyses yourself. Below is the fastest path in.

## TL;DR of the science

- **Robust finding:** dyscalculia = a **noisier number representation** —
  elevated number-encoding noise (≈ +30–35 % on the held first number, n1),
  credible in *every* converged model and both accumulator architectures.
- **Not** a decision-policy deficit (threshold `a` barely moves) and **not** a
  prior/expectation deficit (the observer prior doesn't differ by group).
- **What this paradigm can't settle:** perceptual vs memory noise aren't
  separately identifiable here — they sit on a posterior ridge (r ≈ −0.81) and
  the attribution flips depending on the response model. So: report the noise
  deficit, *not* a perceptual-vs-memory split.
- **The one way forward** (future work / new model): make memory noise act as
  across-trial drift variability (Ratcliff `sv`) while perceptual noise stays
  within-trial — they then have distinct RT-distribution signatures. Motivated
  by an observed misfit (the slow-error QPP wing). See `CONCLUSIONS.md`.

## What to read, in order

1. `notes/CONCLUSIONS.md` — the verdict: what's robust, what's not identifiable,
   why the perceptual/memory attribution flips, and the path forward.
2. `notes/MODELS.md` — the model registry (what each `model-*_trace.nc` is), the
   exact fitting recipe, and the derived quantities behind the figures.
3. `notes/figures/overview_verdict.pdf` — the one figure the conclusions rest on.
4. `notes/ddm_fitting_notes_for_team.md` — if you'll touch convergence: the DDM
   geometry/ridge problem and what to try (start with `dense_mass=True`).

## Getting set up

```bash
git clone git@github.com:ruffgroup/dyscalculic_ddm.git
cd dyscalculic_ddm
conda env create -f environment.yml      # installs the exact bauer (see below)
conda activate dyscalculic_ddm
```

Two things are deliberately **not** in the repo — ask Gilles for both:

- **The behavioural data** (`data/magjudge_behavior_DNumRisk.csv`, human
  subjects) — drop it into `data/`. Everything reads from this file.
- **The posterior traces** (41 GB) — only needed if you want to re-plot rather
  than refit. They're on the cluster:
  `rsync -a sciencecluster:/shares/zne.uzh/gdehol/dyscalculic_ddm/ results/`

The small group-level summary (`notes/data/overview.{json,npz}`) *is* in the
repo, so `python plot_overview.py` regenerates the overview figure without the
traces.

## The exact bauer version (so your fits match mine)

`environment.yml` pins bauer to commit `e406694` (the tip of branch
`feature/rdm-fix-wd-default`, informally "0.4.0"). **Please don't swap in the
`v0.3.0` tag or `main`** — that branch diverged and you'd get subtly different
model code. The pin handles this automatically; just create the env from the
file. (One diagnostics helper used only in methods prose isn't committed
upstream, but no analysis needs it.)

## Reproduce, end to end

```bash
# FIT (cluster GPU):  sbatch --gres=gpu:L4:1 run_fit.sh ddm_group
# AGGREGATE (cluster): python collect_overview.py   -> notes/data/overview.*
# PLOT (local):        python plot_overview.py       -> overview_verdict.pdf
```

Rule of thumb: **fit + aggregate on the cluster, plot + inspect locally.** Most
scripts have a usage block in their top docstring.

Shout if anything doesn't run — happy to walk through it.
