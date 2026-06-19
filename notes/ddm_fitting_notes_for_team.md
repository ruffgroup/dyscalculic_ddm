# DDM fitting — notes for the team working on convergence

Context: the hierarchical magnitude-comparison **DDM** (bauer
`DDMMagnitudeComparisonRegressionModel`, 66 subjects, numpyro/NUTS on a GPU L4)
does not mix reliably. The *static choice* model mixes fine; the **DDM and
sv-DDM do not**. Below is what we've measured and where we think the geometry
problem actually lives, so you don't rediscover it.

## Hard evidence

- **Production DDM fit** (`tune=2000, draws=1000, chains=8, target_accept=0.99`,
  mapjitter init): converges, but slowly and not robustly.
- **Bootstrap recovery** (100 synthetic datasets simulated from the real-data
  posterior at realistic accuracy ≈ 0.85, refit with `tune=800, draws=800,
  chains=4`): **only 21/100 reps reach r̂ ≤ 1.05** (median r̂ = 1.61, max 4.24).
  The same harness on the *choice* model: **98/100 converge** (median r̂ 1.01).
  → The DDM non-convergence is systematic, not a seed lottery, and short
  chains expose it.
- `target_accept` was already **0.99** on the runs that still diverged → step
  size is *not* the lever. This is geometry, not adaptation.
- **More burn-in does NOT help (decisive).** Re-running the 100-rep DDM
  recovery with `tune=2000` (up from 800) changed convergence from 21/100 to
  **24/100** — i.e. negligible. Quadrupling warmup buys nothing. This rules out
  "just tune longer" and confirms the problem is the posterior *geometry* (the
  ridge), not adaptation length. Spend the effort on dense_mass / reparam, not
  on warmup.

## Where the geometry problem is (and is NOT)

### NOT the funnel — it's already non-centered
`build_hierarchical_nodes` (`bauer/core.py:651–654`) is textbook non-centered:
```
offset ~ Normal(0, 1)           # dims=('subject',)
param  = group_mu + group_sd * offset
```
Don't spend time "switching to non-centered" — it's done. (Subject-level
funnel is handled.)

### The actual ridge: the additive noise split
In `shared_perceptual_noise` (`bauer/models/magnitude.py:106`):
```
n1_evidence_sd = perceptual_noise_sd + memory_noise_sd      # first stimulus (encoded + remembered)
n2_evidence_sd = perceptual_noise_sd                        # second stimulus (just encoded)
```
The data constrain `perceptual` (via n2) and the **sum** (via n1). `memory`
only enters as `sum − perceptual` — a difference of two noisy positive
quantities → a strongly **correlated posterior ridge**. Measured directly:

- cross-loading in the (raw) choice recovery: **r ≈ −0.39** between recovered
  perceptual and true memory (and symmetrically);
- σ_perc ↔ σ_mem posterior correlation on the real DDM data: **r ≈ −0.81**.

A −0.8 correlated pair is exactly the diagonal-ridge geometry NUTS handles
worst with a diagonal mass matrix.

## What to try, in priority order

1. **`dense_mass=True` in the numpyro NUTS call** (or block-diagonal over the
   noise group-level nodes). This directly captures the σ_perc↔σ_mem
   correlation we measured. One-liner, highest leverage. Try this *first*.
2. **Reparameterize to the observables.** Regress on `n2_sd` and a positive
   increment `Δ ≥ 0`; derive `perceptual = n2_sd`, `memory = Δ`. This rotates
   the ridge onto a coordinate axis. It's equivalent (bijective) to the
   `independent` / n1n2-separate `memory_model` already in bauer — but NUTS
   sees different geometry, so **benchmark whether `memory_model="independent"`
   mixes better than `"shared_perceptual_noise"` at fixed tune.** Cheap test,
   no new code.
3. **Pathfinder init instead of mapjitter.** `recommended_init='mapjitter'`
   (`bauer/models/ddm.py:144`) seeds *all* chains in one MAP basin; on a ridge
   that's a bad start. A dispersed, curvature-aware multipath init covers it.
   Plug-in point is the init switch at `bauer/core.py:362–403`.
4. **`group_sd` prior.** It's `HalfCauchy(0.25)` (`bauer/core.py:650`) — heavy
   tail. If divergences concentrate at the `*_sd` level, swap to `HalfNormal`
   as a quick experiment.

## Two guardrails — don't regress on these

- **Keep `v_scale` fixed (`fit_v_scale=False`).** There is a *perfect*
  multiplicative `v_scale ↔ evidence_sd` degeneracy in the DDM drift. Freeing
  it reintroduces non-identifiability + divergences. If anyone freed it while
  "adding flexibility," that alone explains a lot.
- **The sv-DDM is ~5 s/it** (8 vectorized chains) because of the across-trial
  `sv` integral. Debug convergence on the **plain DDM first** (much cheaper),
  then port the fix to sv.

## How to judge success

Use the **bootstrap ensemble**, not the single real fit. `recover_bootstrap.py`
already simulated 100 DDM datasets from known parameters at realistic accuracy;
they're on the cluster. A fix that converges on the one real dataset but not
across the ensemble isn't robust. Two metrics, both from that harness:

- **convergence**: fraction of reps with r̂ ≤ 1.05 (currently 21/100 — target
  ≫ 90/100 at `tune≈800`, or document the tune needed);
- **recoverability**: self-recovery slope → 1 and cross-loading → 0 in the
  cross-recovery matrix (`aggregate_recovery.py` → `notes/figures/recovery_cloud.pdf`).
  Convergence alone isn't enough — a fix can converge to a biased/leaky
  estimate. Both must improve.

## Files

- model: `~/git/bauer/bauer/models/ddm.py`, `.../magnitude.py`, `.../core.py`
- fit harness: `fit_ddm.py` (`build_model`), `run_fit.sh`
- recovery harness: `recover_bootstrap.py`, `run_recover.sh`,
  `aggregate_recovery.py`
- single most useful one-liner to try: **`dense_mass=True`**.
