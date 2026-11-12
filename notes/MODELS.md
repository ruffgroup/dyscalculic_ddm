# Models behind `overview_verdict.pdf` — what is what

The single source of truth for the models in
[`figures/overview_verdict.pdf`](figures/overview_verdict.pdf), and nothing
more. The science verdict is in [`CONCLUSIONS.md`](CONCLUSIONS.md). Data:
`data/magjudge_behavior_DNumRisk.csv` (66 subjects, 33 control / 33 dyscalc;
see n1, then n2, choose the larger).

---

## 1. The models in the figure

The figure crosses **four response models** with **two noise decompositions** —
8 fits in all. Every row appears in the overview figure; every column of that
figure is one of these.

| Row in figure | `fit_ddm.py` build | Drift / likelihood |
|---|---|---|
| **Choice** | static cumulative-normal choice | choices only, no RT, no threshold |
| **DDM** | `DDMMagnitudeComparisonRegressionModel` | drift = evidence *difference* (Ratcliff WFPT) |
| **RDM** | advantage race (`RaceDiffusion…`) | per-option accumulators, `w_d≡1` |
| **RDM (+w0/ws)** | RDM with `w_0`, `w_s` freed | adds urgency + magnitude-sum drift |

Each is fit in **two noise decompositions** (§3):

- **shared (perceptual / memory):** `perceptual_noise_sd`, `memory_noise_sd`.
- **independent (n1 / n2):** `n1_evidence_sd`, `n2_evidence_sd`.

Drift terms (RDM): `w_0` = baseline drift / urgency; `w_d` = gain on the evidence
**difference** (fixed = 1); `w_s` = gain on the evidence **sum** (the
overall-magnitude effect). `a` = decision threshold; `t0` = non-decision time.

## 2. The observer prior (the compression story)

The Bayesian observer regresses each percept toward a prior; reliability weight
`w_k = σ_p² / (σ_p² + σ_e,k²)`, compression = `1 − w_k`. The prior is **fixed to
the objective stimulus distribution and shared** across groups and across n1/n2:
`prior_mu = mean(log n1, log n2)`, `prior_sd = σ_p = 0.59`. So any group
difference in compression is carried by the **encoding noise σ_e**, not the
prior. (n1 is weighted below n2 → n1 is compressed more.)

## 3. Group effect & noise decomposition

- **Cell-means group effect.** For each regressed parameter `p`,
  `random_regressors={p: '0 + C(group)'}` → a **separate distribution per
  group**: its own population mean `μ_g` and between-subject SD `τ_g`. The group
  effect is the derived contrast `μ_dyscalc − μ_control`. Coords look like
  `['C(group)[control]', 'C(group)[dyscalc]']`.
- **shared (perceptual/memory):** `n1_evidence_sd = perceptual + memory`,
  `n2_evidence_sd = perceptual` (memory is the extra noise on the held n1).
- **independent (n1/n2):** separate per-stimulus SDs; a bijective
  reparameterization of the above. On LOO the two decompositions are
  indistinguishable (Δelpd < 5).

## 4. The fitting recipe (how every model above is fit)

| Setting | Value | Why |
|---|---|---|
| RT filter | `rt ≥ 0.20 s` | gives the WFPT / Wald likelihood a gradient |
| RT contaminant | `p_outlier = 0.05` | absorbs the slow/fast lapse tail; **the thing that made the 66-subject DDM/RDM converge** (see CLAUDE.md) |
| Race difference gain `w_d` | **fixed = 1** | free → collinear with the noises → no convergence |
| `fit_v_scale` | **False** | perfect `v_scale ↔ evidence_sd` degeneracy if free |
| Group-SD prior | **HalfNormal** | HalfCauchy's fat tail funnels and NUTS stalls |
| Sampler | numpyro, vectorized, 4 chains, tune 2000, draws 1000, L4 GPU | |
| `target_accept` | 0.8 | clean geometry → NUTS default suffices |

Result: r̂ ≈ 1.00, min ESS ~1.5k–3.8k, ~0 divergences for all RDM/choice fits.
(The DDM samples cleanly too but **fits the data poorly** — see CONCLUSIONS.md.)

**Group-level priors, as actually built** (bauer v0.4.0; verified 2026-08-25 by
introspecting `estimation_model.free_RVs` of the two reported specs, not read
off the source). Everything below is on the **pre-transform (link) scale**;
`a`/`t0`/`w_0`/`perceptual_noise_sd`/`memory_noise_sd` are softplus-transformed
after the design-matrix product, `w_s` has an identity link.

| Node | Prior in the reported (cell-means) fits |
|---|---|
| `{p}_mu` (population mean, per group) | `Normal(0, 1)` for **every** parameter and both groups |
| `{p}_sd` (between-subject τ, per group) | `HalfNormal(σ = [0.25 control, 0.10 dyscalc])` for `perceptual_noise_sd` / `memory_noise_sd`; `HalfNormal([0.25, 0.25])` for `a`, `t0`, `w_0`, `w_s` |
| `{p}_offset` | `Normal(0, 1)` |

Two traps this table exists to prevent:

1. **Half-normal, not half-Cauchy.** `BaseModel.group_sd_dist` defaults to
   `'halfnormal'` from bauer 0.3.0 on, and `_group_sd_rv` builds
   `pm.HalfNormal(name, sigma=scale)`. The `--halfnormal` flag in `fit_ddm.py`
   only re-asserts the default. Any methods text citing Gelman (2006) for a
   half-Cauchy group-SD prior is describing the pre-0.3.0 behaviour, not these
   fits.
2. **The `mu_intercept` / `sigma_intercept` values are NOT the priors here.**
   `RaceMixin.get_free_parameters()` sets `a` → 0.54, `t0` → −1.51,
   `w_0` → 2.41 with `sigma_intercept` 0.3/0.5/0.5, and the magnitude front-end
   sets `mu_intercept` −1 for both noise SDs. The regression path
   (`RegressionMixin.build_hierarchical_nodes`) applies those to column 0
   **only when that column is literally named `Intercept`**. With
   `random_regressors={p: '0 + C(group)'}` the columns are
   `C(group)[control]` / `C(group)[dyscalc]`, so the branch never fires and all
   cells fall back to `Normal(0, sigma_regressors=1)`.

The same "column 0 is the intercept" assumption explains the asymmetric τ
prior: `cauchy_sigma[0]` is unconditionally set to `cauchy_sigma_intercept`
(0.25), while later columns get `cauchy_sigma_regressors`, which the magnitude
front-end tightens to 0.1 for the two noise SDs (`TIGHT_SLOPE`). Control
inherits the intercept scale, dyscalc the slope scale. It is an artifact of the
parameterization, not a deliberate choice, and it is worth knowing that the
posterior τ for memory noise (control 0.65, dyscalc 0.35, P = 0.03 in
`supp_rdm_params` row 3) is the one result that sits closest to it. Group
**means**, and therefore every headline contrast, are on symmetric priors.

**Trial exclusion accounting.** The task gives no "too late" or
correct/incorrect feedback of any kind (checked `trial_magJudge.py` end to
end). Trials with no recorded response aren't logged with a missing RT —
they're simply absent: `numrisk/behavior_magjudge/utils.py::get_data()`
computes `rt = onset(choice) − onset(stimulus 2)`, which is `NaN` when no
key was pressed, and an unqualified `df.dropna()` there (commented only as
"removes subs without group assignment") silently drops those rows too. Of
the full 3-run design (66 participants × 180 trials = 11,880), **212 trials
(1.8%) have no logged response** and are already gone before this project's
CSV exists (per-subject counts run 149–180; no subject near a round
fraction like ~120, so this reads as scattered single-trial omissions, not
dropped runs). The `rt ≥ 0.20 s` filter above removes a further 3 trials,
leaving the 11,665 analysed. The `p_outlier` contaminant mixture is a
separate mechanism — it reweights trials that *do* have a recorded
`(rt, choice)` (occasional fast guesses/lapses); it cannot and does not
touch these 212 trials, since there's no `rt` for either likelihood term to
evaluate.

## 5. The 10 pages of `overview_verdict.pdf`

1–2. Parameter contrasts (dyscalc−ctrl) per model, shared and n1/n2
decompositions; gray = 94% HDI ∋ 0, number = P(Δ>0).
3. Raw n1/n2 encoding SD per group (natural units).
4. Compression driver: reliability weights `w` and SD log-ratio, with Δgap.
5. Raw perceptual/memory SD per group (natural units).
6. Between-subject SD (τ) per parameter and group.
7. Psychometric PPC by reference magnitude n1 (DDM & RDM).
8. RT PPC by n1 (DDM & RDM) — the DDM misfit is visible here.
9. PPCs: P(chose n2), RT×difficulty, RT×magnitude (DDM vs RDM).
10. LOO model comparison (RDM ≫ DDM; noise decomposition ≈ tie).

Derived quantities are computed by `collect_overview.py` into
`data/overview.{json,npz}` and rendered by `plot_overview.py`.
