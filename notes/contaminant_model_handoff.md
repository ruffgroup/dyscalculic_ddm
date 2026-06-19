# DDM/RDM contaminant (`p_outlier`) — handover for the dyscalculic team

What changed in bauer (branch **`feature/ddm-lapse`**), why, and what you need to
do. Bottom line: the `p_outlier` lapse contaminant is now **built into every DDM
/ RDM model by default** (HSSM-style), and it's what makes the 66-subject DDM
actually converge.

## ⚠️ Breaking change — migrate your imports
The separate `*Lapse*` DDM/RDM classes are **removed**. The contaminant is now
in the base models. Migrate:

| old (removed) | new |
|---|---|
| `DDMMagnitudeComparisonLapseRegressionModel` | `DDMMagnitudeComparisonRegressionModel` |
| `DDMMagnitudeComparisonLapseModel` | `DDMMagnitudeComparisonModel` |
| `DDMRiskLapseRegressionModel` | `DDMRiskRegressionModel` |
| `RaceDiffusion*LapseModel` / `*LapseRegressionModel` | the base `RaceDiffusion*` class |

The base classes now carry `p_outlier=0.05` by default — so just dropping the
`Lapse` from the name gives you the contaminant model. (The **static-choice**
`*Lapse*` models — `RiskLapseModel`, `MagnitudeComparisonLapseModel`, etc. — are
**unchanged**; there the lapse is genuinely optional.)

## How to get it
```bash
cd ~/git/bauer            # the clone your env pip-installed -e
git fetch origin feature/ddm-lapse
git checkout feature/ddm-lapse      # editable install picks it up; no reinstall
python -c "from bauer.models import DDMMagnitudeComparisonRegressionModel as M; \
           print('p_outlier default:', M.p_outlier)"   # -> 0.05
```

## What the contaminant is (HSSM convention)
On a fraction `p_outlier` of trials the response is a contaminant (fast guess /
attention lapse) from a flat density over RT, not the diffusion process. The
per-trial log-likelihood is the stable mixture:
```
logp = log( (1 - p_outlier)*exp(ll_WFPT) + p_outlier*exp(lapse_logp) + 1e-29 )
```
Contaminant density = `Uniform(0, lapse_upper)` (default 20 s) × 50/50 choice, so
`lapse_logp = log(0.5) - log(lapse_upper)` (constant). Configurable via
`lapse_upper`, `lapse_choice_5050`.

## Controlling `p_outlier` (it's an attribute on the model)
- `model.p_outlier = 0.05` — **default, fixed** (HSSM default). Not a sampled
  parameter, so it can't funnel.
- `model.p_outlier = 0.0` — **pure WFPT** (recovers the old contaminant-free DDM).
- `model.p_outlier = 'hierarchical'` — per-subject estimated rate (Beta "1/x"
  group prior). **Avoid for the DDM** — weakly identified in the WFPT mixture; it
  diverges 4000/4000 even with seeded Pathfinder. Use the fixed scalar; if you
  must estimate, estimate a single scalar (not per-subject).
Set the attribute before `build_estimation_model` (or re-run
`model.free_parameters = model.get_free_parameters()` after changing it).

## simulate / PPC (matches HSSM)
`simulate`/`ppc` **generate** the contaminant by default (`simulate_contaminant
= True`) — matching HSSM's posterior predictive (`rng_fn` → `_apply_lapse_model`).
So a PPC overlaid on real data covers the RT tail. Set
`model.simulate_contaminant = False` for the clean decision process only.

## Why this matters — it fixes 66-subject convergence
The whole point. On the real dyscalculic data:

| 66-subj DDM | r̂ | divergences | min ESS | time |
|---|---|---|---|---|
| plain, no contaminant (HalfNormal SD) | 1.53 | 0 | 33 | 43 min |
| plain, no contaminant (HalfCauchy SD) | 2.42 | 0 | 5 | 74 min |
| **+ fixed `p_outlier=0.05` (mapjitter)** | **1.00** | **0** | **4264** | 51 min |

The plain DDM doesn't converge at scale — the slow-tail RTs bump the likelihood.
The fixed contaminant absorbs them → r̂ 1.00, ESS 4264. Same disease as the
risk model (an unmodeled lapse), here on RT.

## Production recipe
`DDMMagnitudeComparisonRegressionModel` (contaminant on by default),
`memory_model='shared_perceptual_noise'`, `fit_v_scale=False`, **mapjitter**
init, `tune=2000`, `target_accept=0.99` (→ ~0.9 for ~2.3× speed once converged;
a few divergences return). Notes:
- **Pathfinder NOT needed** with the contaminant — mapjitter converges
  identically (r̂ 1.00, ESS 4264 vs 3732). Keep `find_init='pathfinder'` (opt-in,
  `pymc_extras`) as break-glass for genuinely nasty geometries; it's expensive
  (`find_MAP` + multipath VI) and seeds from the MAP.
- `group_sd_dist='halfnormal'` (opt-in) tames the HalfCauchy group-SD funnel;
  helps the contaminant-free case but unnecessary once the contaminant is in.

## Guardrails (unchanged)
Keep `v_scale` fixed (`fit_v_scale=False`); filter `rt < 0.20 s`; debug the plain
DDM before the sv-DDM.

## Pointers
- Full guide: `bauer/notes/sampling_robustness.md` (branch `feature/ddm-lapse`).
- Debug ladder + driver: `bauer/examples/dyscalculic_ddm_debug/`.
- Results: `~/ddm_debug/*.json` on the cluster.
