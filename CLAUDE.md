# CLAUDE.md — dyscalculic_ddm

Project context for Claude Code sessions. Read alongside `README.md` and the
`notes/` directory (the real source of truth for the science).

## What this project is

Bayesian-observer + sequential-sampling modelling of a symbolic
magnitude-comparison task (DNumRisk: 66 subjects, 33 control / 33 dyscalculic,
~11.7k trials; "see n1, then n2, choose the larger"). The decision-maker
repeatedly samples a perceptual posterior each step, so noisy perception lands
in the drift rate. Response layers: static **choice**, **DDM** (drift = evidence
difference), and **advantage RDM** (van Ravenzwaaij 2020; per-option
accumulators, `w_0`/`w_d`/`w_s`). Everything lands in
`notes/figures/overview_verdict.pdf`.

## The result (current — supersedes any older "ridge / not identifiable" notes)

The **RDM** is the model to trust; the **DDM** does not fit (can't produce the
magnitude effect; loses LOO by ~730, `bad-k 87`). On the RDM, dyscalculics show:

- **A noisier number representation** (P = 1.00): perceptual noise 0.34 → 0.45;
  **memory noise roughly doubles, 0.049 → 0.091** (the held first number is hit
  hardest).
- **A compensatory threshold increase** (`a` 0.50 → 0.65, P = 1.00) → they're
  clearly slower but only slightly less accurate (calibrate-to-target-accuracy).
- **Higher urgency** (`w_0` 0.31 → 0.40), a **slightly larger magnitude effect**
  (`w_s`), and a **larger compression effect**.

Headline: **noisier → slower, with stronger (esp. memory) compression.** This is
robust and is an improvement on the earlier draft. Do **not** repeat the old
"perceptual vs memory is an unidentifiable ridge" framing — with the proper RDM
the encoding split is informative (memory carries the bigger hit). Full numbers
in `notes/CONCLUSIONS.md`; models in `notes/MODELS.md`.

## Hard-won methods lessons (save these for any future SSM project)

These three took real time to find and each has a known fix:

1. **The likelihood is brutally sensitive to contaminants** (DDM especially,
   RDM too). A *single* trial out of ~12,000 — one impossibly-easy trial answered
   wrong, or an RT shorter than the true `t0` — creates a ridge that stops NUTS
   converging. (This is why Ratcliff et al. fit quantiles rather than the raw
   likelihood.) **Fix:** a 5 % `p_outlier` contaminant — 5 % chance of a uniform
   random response at a uniform random time. Same idea as Teulincx & Ratcliff
   and as Michael Frank's hssm. (bauer's likelihoods come straight from **hssm**,
   now an optional dependency.)
2. **Use HalfNormal, not HalfCauchy, on the group SDs.** HalfCauchy's fat tail
   allows unrealistically large SDs → an extreme funnel between group SD and
   subject parameters that even NUTS can't traverse. Gelman moved off HalfCauchy
   to HalfNormal for exactly this; so did we.
3. **Fix `w_d = 1`.** Left free it's highly (not perfectly) collinear with the
   noise SDs and won't converge — and if you fix n1/n2 SD but free `w_d`, the
   model just finds a *slower* `w_d` for dyscalculics, which is nearly the same
   thing as more noise per sample. Also keep `fit_v_scale=False` (perfect
   `v_scale ↔ evidence_sd` degeneracy).

## bauer version — use exactly v0.4.0

The models live in **bauer**. The analyses were run with the immutable tag
**`v0.4.0`** = `github.com/ruffgroup/bauer` @ `e406694` (tip of
`feature/rdm-fix-wd-default`). `environment.yml` pins `@v0.4.0`.

- ❌ Do **not** use `v0.3.0` or `main` — the 0.4.0 line is on a branch that
  *diverged* from the release line; anything else is subtly different model code.
- Quirk: the in-tree `__version__` on this commit still reads `0.3.0`; the
  **tag** is authoritative.

## Where things live

- **Input data** (`data/*.csv`): human-subjects, **gitignored**. Real input is
  `data/magjudge_behavior_DNumRisk.csv`; everything reads from it.
- **Traces** (`results/`, 41 GB): **gitignored**. On the cluster at
  `/shares/zne.uzh/gdehol/dyscalculic_ddm/model-<label>_trace.nc`.
- **Committed summary**: `notes/data/overview.{json,npz}` — lets the overview
  figure regenerate locally with `plot_overview.py`, no traces needed.

## Cluster vs local (matches the global house rule)

- **Fit + aggregate → cluster** (GPU L4, `bauer_cuda` env + the v0.4.0 bauer):
  `fit_ddm.py` via `run_fit.sh`, then `collect_overview.py`.
- **Plot + inspect → local**: `plot_overview.py` (and the recovery/diagnostic
  scripts) read the small summaries and emit PDFs into `notes/figures/`.

## Conventions

- `fit_ddm.py` holds `build_model(spec, df)` — a spec mini-language
  (`arch:param1,param2`, e.g. `rdm_nowd:perceptual_noise_sd,memory_noise_sd,a,t0`)
  and the **cell-means** group parameterization (`random_regressors={p:'0 + C(group)'}`).
  See `notes/MODELS.md §3–4`.
- Most scripts have a runnable usage block in the top docstring — check it.
- The repo is kept **lean**: only the figure pipeline (`fit_ddm`,
  `collect_overview`, `plot_overview`, `figstyle`), the recovery harness
  (`recover_bootstrap`, `aggregate_recovery`, `run_recover`), and the trace
  diagnostics (`summarize_traces`, `group_contrasts`, `chain_diag`, `ridge_diag`).
  Don't reintroduce one-off make_*/plot_* scripts; they live in git history.

## Git

This repo is **public** at `github.com/ruffgroup/dyscalculic_ddm`. Keep
human-subjects CSVs and the 41 GB traces out of commits (both are gitignored).
End commit messages with the standard co-author trailer.
