# CLAUDE.md — dyscalculic_ddm

Project context for Claude Code sessions. Read alongside `README.md` and the
`notes/` directory (the real source of truth for the science).

## What this project is

Bayesian-observer + sequential-sampling modelling of a symbolic
magnitude-comparison task (DNumRisk: 66 subjects, 33 control / 33 dyscalculic,
~11.7k trials; "see n1, then n2, choose the larger"). Three response layers —
static **choice**, **DDM**, **RDM** — over a shared bauer cognitive front-end,
crossed with which parameters vary by group, to localise the dyscalculia
deficit.

**Settled finding:** dyscalculia = elevated number-encoding noise (~+30–35 % on
n1), not a decision-policy or prior deficit. **Open / non-identifiable:**
perceptual vs memory noise (a posterior ridge, r ≈ −0.81). Don't claim the
perceptual/memory split is resolved. Details in `notes/CONCLUSIONS.md` and
`notes/MODELS.md`.

## The bauer version trap (read before touching the env or refitting)

There are **three** bauer branches in play locally; only one is "the science":

| Branch | Commit | Role |
|---|---|---|
| **`feature/rdm-fix-wd-default`** | **`e406694`** ("0.4.0", `__version__` 0.3.0) | ✅ **THE science version.** Every overview/registry model in `notes/MODELS.md` was fit with this (deployed on the cluster as `~/bauer-040` via `PYTHONPATH`). `environment.yml` pins this commit. |
| `sv-memory-noise` | `600d668` (`__version__` 0.2.0) | Earlier laptop smoke-fits + recovery work. NOT the registry models. |
| `feature/ddm-lapse` (`~/bauer-lapse`) | — | Superseded/invalid: lacked the `fit_w_d` toggle, so "w_d-fixed" runs there were silently w_d-free. Ignore. |

- `feature/rdm-fix-wd-default` ≠ the `v0.3.0` tag — they **diverged**. Do not
  pin the tag or `main`; pin the commit (already done in `environment.yml`).
- `bauer.utils.diagnostics.trim_stuck_chains` is referenced in the methods prose
  but is **uncommitted upstream** and **unused by any analysis script** here.
  If you commit it to bauer, do it on `feature/rdm-fix-wd-default`.

## Where things live

- **Input data** (`data/*.csv`): human-subjects, **gitignored**. The real input
  is `data/magjudge_behavior_DNumRisk.csv`; everything reads from it.
- **Traces** (`results/`, 41 GB): **gitignored**. On the cluster at
  `/shares/zne.uzh/gdehol/dyscalculic_ddm/model-<label>_trace.nc`.
- **Committed summary**: `notes/data/overview.{json,npz}` (small, group-level) —
  lets the overview figure regenerate locally without the traces.
- **Figures**: `notes/figures/` (committed).

## Cluster vs local (matches the global house rule)

- **Fit + aggregate → cluster** (GPU L4, `bauer_cuda` env + `PYTHONPATH` to the
  bauer-040 checkout): `fit_ddm.py` via `run_fit.sh`, then `collect_overview.py`.
- **Plot + inspect → local**: `plot_overview.py`, `make_*.py`, `plot_*.py` read
  the small summaries / traces and emit PDFs in `notes/figures/`.

## Conventions

- `fit_ddm.py` holds `build_model(spec, df)` — a spec mini-language
  (`arch:param1,param2`, e.g. `rdm_nowd:perceptual_noise_sd,memory_noise_sd,a,t0`)
  and the **cell-means** group parameterization (`random_regressors={p:'0 + C(group)'}`,
  separate per-group mean AND between-subject SD). See `notes/MODELS.md §3`.
- Standard fitting recipe (RT filter ≥ 0.20 s, `fit_v_scale=False`, `w_d` fixed,
  `p_outlier=0.05`, halfnormal group-SD prior, numpyro vectorized) is in
  `notes/MODELS.md §1`. Don't free `v_scale` (perfect degeneracy with
  `evidence_sd`).
- Most scripts have a runnable usage block in the top docstring — check it.

## Git

This repo is **public** at `github.com/ruffgroup/dyscalculic_ddm`. Keep
human-subjects CSVs and the 41 GB traces out of commits (both are gitignored).
End commit messages with the standard co-author trailer.
