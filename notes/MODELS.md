# Dyscalculia magnitude-comparison models — what is what

Single source of truth for the models behind `notes/figures/overview_verdict.pdf`
and the model-comparison work. Data: `data/magjudge_behavior_DNumRisk.csv`
(66 subjects: 33 control / 33 dyscalc; see n1 then n2, choose larger).

---

## 1. The standard recipe (everything in the overview figure uses this)

All "science" models are fit with **bauer 0.4.0** (branch `feature/rdm-fix-wd-default`,
deployed on the cluster as `~/bauer-040`, used via `PYTHONPATH`). Settings:

| Setting | Value | Why |
|---|---|---|
| RT filter | `rt >= 0.20 s` | gives the WFPT/Wald likelihood gradient |
| Race drift gain `w_d` | **fixed = 1** (arch `rdm_nowd`) | w_d is the v_scale analog; freeing it = ridge with σ_e (r≈0.83) that wrecks mixing |
| `fit_v_scale` | **False** (always) | perfect v_scale↔evidence_sd degeneracy if free |
| RT contaminant `p_outlier` | **fixed 0.05** | absorbs the slow-RT lapse tail; the thing that made the 66-subj DDM converge |
| Group-SD prior `group_sd_dist` | **halfnormal** (`--halfnormal`) | HalfCauchy tail funnels; HalfNormal tames it |
| `target_accept` | **0.8** | clean geometry → NUTS default suffices; 0.99 was a crutch (now `run_fit.sh` default) |
| Sampler | numpyro, vectorized, 4 chains, tune 2000, draws 1000, L4 GPU | |
| Group effect | **cell-means** (see §3) | separate per-group mean AND SD; correct for a between-subjects factor |

Result: r̂ ≈ 1.00, min ESS ~1.3k–3.8k, ~0 divergences for every model below.

---

## 2. The observer prior (READ THIS for the compression story)

The Bayesian observer regresses each percept toward a prior; reliability weight
`w_k = σ_p² / (σ_p² + σ_e,k²)`, compression = `1 − w_k`.

- **By default `fit_prior=False`**: the prior is **fixed to the objective stimulus
  distribution and is the SAME for both groups and for n1 and n2**:
  `prior_mu = mean(log n1, log n2)`, `prior_sd = σ_p = (std(log n1)+std(log n2))/2 = 0.587`.
- Consequence: any group difference in compression is **forced onto the encoding
  noise σ_e**, because σ_p is pinned. σ_e and σ_p are partially degenerate in `w`
  (compression depends on the ratio σ_e/σ_p).
- **`rdm_n1n2_priorgrp`** (below) frees `prior_mu` and `prior_sd` per group to test
  whether the noise deficit survives or partly reallocates to a prior difference.

---

## 3. Group-effect parameterization (cell-means)

`fit_ddm.build_model` default (since 2026-06-10): for every regressed parameter `p`,
`random_regressors={p: '0 + C(group)'}` → bauer auto-promotes it to the fixed design,
giving a **separate distribution per group**: own population mean `μ_g` AND own
between-subject SD `τ_g`. No random-slope-on-contrast confound, no forced
homoscedasticity. The group effect is the derived contrast `μ_dyscalc − μ_control`.

- Coords look like `['C(group)[control]', 'C(group)[dyscalc]']`.
- `GROUP_PARAM=legacy` env var → old `regressors={p:'group'}` (Intercept + random
  slope on the contrast — deprecated, do not use for the science).

Noise decompositions (set by which params are named):
- **shared** (`perceptual_noise_sd`, `memory_noise_sd`): `n1_evidence_sd = perceptual + memory`,
  `n2_evidence_sd = perceptual` (memory is the extra noise on the held n1).
- **independent / n1n2** (`n1_evidence_sd`, `n2_evidence_sd`): separate per-stimulus SDs.

---

## 4. Model registry (the overview-figure set)

All fit with the §1 recipe, cell-means, `bauer-040`. Trace files in
`/shares/zne.uzh/gdehol/dyscalculic_ddm/model-<label>_trace.nc`.

| tag (in figure) | build spec | trace label | regressed on group |
|---|---|---|---|
| Choice | `choice:perceptual_noise_sd,memory_noise_sd` | `choice_pm_hn` | perc, mem |
| DDM | `ddm:perceptual_noise_sd,memory_noise_sd,a,t0` | `ddm_full_cont_hn` | perc, mem, a, t0 |
| RDM | `rdm_nowd:perceptual_noise_sd,memory_noise_sd,a,t0` | `rdm_full_cont_hn` | perc, mem, a, t0 |
| Choice (n1/n2) | `choice:n1_evidence_sd,n2_evidence_sd` | `choice_n1n2_hn` | n1, n2 |
| DDM (n1/n2) | `ddm:n1_evidence_sd,n2_evidence_sd,a,t0` | `ddm_n1n2_full_cont_hn` | n1, n2, a, t0 |
| RDM (n1/n2) | `rdm_nowd:n1_evidence_sd,n2_evidence_sd,a,t0` | `rdm_n1n2_full_cont_hn` | n1, n2, a, t0 |
| RDM (+w0/ws) | `rdm_nowd:perceptual_noise_sd,memory_noise_sd,a,t0,w_0,w_s` | `rdm_max_cont_hn` | + w0, ws |
| RDM (+w0/ws, n1/n2) | `rdm_nowd:n1_evidence_sd,n2_evidence_sd,a,t0,w_0,w_s` | `rdm_n1n2_max_cont_hn` | + w0, ws |
| **RDM (prior~group)** | `rdm_nowd:n1_evidence_sd,n2_evidence_sd,a,t0,prior_mu,prior_sd` | `rdm_n1n2_priorgrp_cont_hn` | n1, n2, a, t0, **prior_mu, prior_sd** (fit_prior auto-on) |

`a` = decision threshold; `t0` = non-decision time; `w_0` = baseline (common) drift;
`w_s` = overall-magnitude (sum) drift term; `w_d` ≡ 1 (fixed).

### Convergence / robustness experiments (NOT science models — diagnostics)
- `rdm_wdfix_{cont,nocont}_hn[_s1..s3]` — 4-seed × contaminant-on/off replicates that
  showed the contaminant **helps** the RDM (≈0 vs 32–130 divergences) and w_d-fix gives
  ESS ~3700. `*_ta80` = ta=0.8 vs 0.99 A/B.
- `*_cont` (HalfCauchy) and the original `rdm_pm/rate/all3` — pre-cell-means, pre-w_d-fix;
  superseded. Earlier w_d "fixed" runs on `bauer-lapse` were **invalid** (that branch
  lacked the `fit_w_d` toggle, so w_d was silently free).

---

## 5. Derived quantities (computed in `collect_overview.py`, plotted by `plot_overview.py`)

Per model, stored in `notes/data/overview.json`:
- `contrasts[p]` — dyscalc − control of the cell means (link scale) + P(Δ>0).
- `raw[p]` — per-group value in **natural units** (transform of μ_g; softplus for SDs).
- `group_sds[p]` — per-group between-subject SD τ_control, τ_dyscalc + the τ contrast.
- `asymmetry` (n1/n2 models only), all per draw with σ_p = 0.587 (fixed):
  - `diff`  = σ_n1 − σ_n2 (SD difference, scale-dependent)
  - `logratio` = log(σ_n1/σ_n2) (scale-invariant)
  - `w1`,`w2`,`gap` = reliability weights `σ_p²/(σ_p²+σ_e²)` and `gap = w2 − w1`
    — the **correct, variance-based, prior-aware driver of differential compression**.
  - contrasts (dyscalc − control) for diff / logratio / gap with P(>0).

### Overview figure pages (`overview_verdict.pdf`)
1–2 parameter contrasts (perc/mem; n1/n2), gray = 94% HDI ∋ 0 · 3 raw σ_n1/σ_n2 ·
4 compression driver (reliability weights + SD log-ratio) · 5 raw perc/mem ·
6 per-group SDs · 7 psychometric by n1 (DDM+RDM) · 8 RT by n1 (DDM+RDM) ·
9 PPCs (compression/difficulty/magnitude) · 10 LOO.

---

## 6. Scripts

| script | role | where it runs |
|---|---|---|
| `fit_ddm.py` | `build_model(spec, df)` + fit; the spec mini-language and cell-means default | cluster GPU (`run_fit.sh`) |
| `run_fit.sh` | SLURM wrapper; `--target-accept 0.8` default; explicit `--out` overrides the auto name | cluster |
| `collect_overview.py` | aggregate traces → `notes/data/overview.{json,npz}` (contrasts, raw, SDs, asymmetry, PPC bins) | cluster (`PYTHONPATH=bauer-040`) |
| `plot_overview.py` | render `overview_verdict.pdf` from the summary | local (house style) |
| `summarize_traces.py`, `group_contrasts.py`, `chain_diag.py`, `ridge_diag.py` | convergence / contrast / ridge diagnostics | cluster |

## 7. Key empirical conclusions (so far)
- Deficit = **elevated encoding noise on the held number (n1)**; ~+30–35% in natural units.
- n1 weighted below n2 → n1 compressed more; in dyscalc both weights drop AND the
  gap w2−w1 widens → more differential compression (P(gap_dys>gap_ctrl) 0.80–0.93).
- Threshold `a`, t0, w0/ws shifts appear but are degenerate (scale ridge) — not separately
  identified; report the noise deficit, not the threshold/drift split.
- RDM dominates DDM on LOO (Δ≈730, reliable); noise decomposition is a reparameterization.
- **Open check:** does the noise deficit survive when the prior is freed per group
  (`rdm_n1n2_priorgrp`)? — running.
