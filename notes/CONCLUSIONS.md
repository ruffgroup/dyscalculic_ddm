# Dyscalculia magnitude-comparison modelling — conclusions

Data: DNumRisk symbolic magnitude comparison, 66 subjects (33 control / 33
dyscalculic), ~11.7k trials ("see n1, then n2, choose the larger"). Everything
here is the dyscalc−control story summarised in
[`figures/overview_verdict.pdf`](figures/overview_verdict.pdf); the model
registry and fitting recipe are in [`MODELS.md`](MODELS.md).

## The model (one paragraph)

The decision-maker repeatedly samples a **perceptual posterior** at every time
step. Biased / noisy perception therefore lands in the **drift rate**: hard
trials (small evidence difference) are slower. We fit two response layers on top
of the same Bayesian-observer front-end (bauer):

- **DDM** (Ratcliff 1978; Ratcliff & McKoon 2008) — evidence is the *difference*
  of the two repeatedly-sampled posterior means. No extra trial-to-trial
  variability beyond the sampling mechanism itself.
- **Advantage RDM** (van Ravenzwaaij et al. 2020) — each option has its own
  accumulator. Drift = `w_0` (intercept → urgency) + `w_d`·(evidence
  *difference*) + `w_s`·(evidence *sum*). `w_d ≡ 1` (fixed; see MODELS.md). The
  `w_s`·sum term captures the **overall-magnitude effect**: people are faster
  for larger numerosities, over and above difficulty.

**The RDM is the model to trust; the DDM is not.** The DDM cannot produce the
magnitude effect (flat where the data slope — `overview_verdict.pdf` p8–9), and
on LOO the RDM beats it by **Δelpd ≈ 730** with reliable Pareto-k, whereas the
DDM is `bad-k 87 → LOO unreliable` (p10). Among RDM variants the noise
decomposition (perceptual/memory vs n1/n2) barely changes fit (Δelpd < 5) — it
is a reparameterization, not a better model.

## The result — dyscalculia is a noisier number representation

All numbers below are the RDM (natural units, dyscalc vs control; P = P(Δ>0)
from `overview_verdict.pdf` p1, p5).

1. **Elevated encoding noise — the headline, and it is robust (P = 1.00).**
   - Perceptual noise SD: **0.34 → 0.45**.
   - Memory noise SD: **0.049 → 0.091 — it roughly doubles.** They are generally
     much noisier, but *even more so for the held-in-memory first number.*
2. **A compensatory threshold increase (P = 1.00).** Decision threshold
   `a`: **0.50 → 0.65**. Reduced effective drift would, alone, make them both
   slower *and* much less accurate; instead they raise the threshold (the
   well-known calibrate-to-a-target-accuracy strategy), so they end up **clearly
   slower but only slightly less accurate** — exactly the raw behaviour.
3. **Higher urgency (P = 0.99).** Baseline drift `w_0`: **0.31 → 0.40** — as if
   more worried about not responding in time.
4. **A slightly larger magnitude effect (P = 0.91).** Sum term `w_s`:
   **0.063 → 0.087**.
5. **A larger compression effect.** Lower reliability weights
   `w = σ_p²/(σ_p²+σ_e²)` and a wider `w₂−w₁` gap → more regression to the prior,
   more so in dyscalc (Δgap P = 0.80–0.93, p4). Compression rises because the
   representation is noisier.
6. **Non-decision time `t0` does not differ** (0.38 vs 0.36) and the **observer
   prior is shared** (fixed to the stimulus distribution, σ_p = 0.59).

**Model-free behaviour it explains.** Dyscalculics are slightly slower and
slightly less accurate. That smells like reduced drift — but the threshold
compensation is why the accuracy cost stays small while RT clearly rises. The
RDM PPCs reproduce the psychometric curves, the RT×difficulty slope, and the
RT×magnitude slope for both groups (p7–9).

## Recoverability (`figures/recovery_cloud.pdf`)

Cross-recovery on 100 synthetic datasets simulated at realistic accuracy
(≈ 0.85), refit, comparing true vs recovered perceptual/memory Δ:

- **Static choice:** converges easily (98/100) but perceptual and memory Δ
  **leak into each other** (off-diagonal r ≈ −0.4) — choices alone can't cleanly
  separate the two encoding components.
- **DDM:** converges in only **24/100** refits and recovers the split poorly —
  another reason not to lean on the DDM.

The clean recoverability check for the **RDM** perceptual/memory split (the model
we actually report) is the recommended next analysis; `recover_bootstrap.py` +
`aggregate_recovery.py` is the harness for it.

## Bottom line

Dyscalculia = a **noisier number representation** — most sharply an elevated
*memory* noise on the held first number (≈ doubled) — which (i) slows responses,
(ii) is compensated by a higher decision threshold so accuracy drops only
slightly, and (iii) produces stronger regression-to-the-prior compression. The
RDM, with its urgency (`w_0`) and magnitude (`w_s`) terms, fits this cleanly;
the DDM does not.
