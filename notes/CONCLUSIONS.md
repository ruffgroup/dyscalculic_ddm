# Dyscalculia magnitude-comparison modelling — conclusions

Data: DNumRisk symbolic magnitude comparison, 66 subjects (33 control / 33
dyscalc), ~11.7k trials. Models: Bayesian-observer front-end (bauer) with three
response layers — static **choice**, **DDM**, **RDM** — crossed with which
parameters vary by group; fit on GPU (numpyro, tune/draws 4000, 8 chains where
needed). Convergence judged at the group level; a single stuck chain handled by
`bauer.utils.diagnostics.trim_stuck_chains` (see methods note).

## What is robust (report these)

1. **Behaviour (model-free).** Dyscalculics are **slower** (median RT +~0.15 s),
   slightly **less accurate** (~−0.02), and show a **steeper magnitude /
   compression effect** (P(chose n2) rises more steeply with n1).
2. **Elevated number-encoding noise in dyscalculics** — credible in *every*
   converged model and both accumulator architectures. This is the headline.
3. **Not a decision-policy or expectation deficit.** Threshold `a` barely moves
   (precise near-null in RDM, weak trend in DDM); the prior (`μ_p`, `σ_p`) does
   **not** differ by group (all-free fits). It is a **noise** effect, as expected.
4. **The compression is the regression-to-the-prior signature**, and in this
   regime (encoding SD ≈ 0.2 ≪ prior SD ≈ 0.6) raising *either* perceptual or
   memory noise widens the reliability gap `w₂−w₁` and **increases**
   compression — so it does not, by itself, single out a noise component.

## What is NOT identifiable from this design

**Perceptual vs memory noise cannot be separated here.** Evidence:
- The two are collinear in the joint fit (`r = −0.81`).
- The attribution **flips with the response model**: choice-only → memory↑
  (P=1.00); DDM/RDM → perceptual↑ (P≈0.97–1.00). With *both* on the corrected
  KLW normalizer the split persists, so it is not the normalization bug (below).
- **Either noise alone reproduces RT + accuracy + compression** about equally
  (single-noise PPCs match the data within error).
- ISI does **not** modulate the effect (known from prior work), so the
  delay handle that would separate memory from perceptual is absent.
- **Parameter recovery:**
  - *Choice model (done, cross-model — fit to DDM-generated data):* returns
    **memory↑ regardless of the generating truth** (perceptual-truth +0.26 →
    recovered memory +0.39, P=0.96, perceptual null; memory-truth → memory
    +0.67, P=1.00). So the static-choice "memory" verdict is **uninformative**
    about which encoding component moved — it reports memory either way.
  - *DDM within-model recovery (PENDING, cluster `recover_ponly`/`recover_monly`):*
    the decisive test — does the DDM recover perceptual-truth→perceptual and
    memory-truth→memory? Verdict to be inserted. This, not verbal reasoning,
    settles whether the perceptual↔memory split is real or a ridge.

Bottom line: **dyscalculia = noisier number representation; which encoding
component (perceptual vs memory) is underdetermined by this paradigm.**

## Why the split flips (mechanism, understood)

Choices carry the *compression*, which the static observer can only produce via
the memory (asymmetry) term → choice→memory. The accumulator models also fit
**RT**, which credibly supports raising the common (perceptual) term; given the
collinearity, the joint DDM/RDM then credits perceptual and squeezes memory to
null. Both are "correct" within their likelihood — it is a ridge, not a
dissociation.

## Methods findings worth keeping

- **Static-choice noise normalization bug** (`notes/choice_vs_ddm_decision_noise_inconsistency.md`):
  the static model used raw `evidence_sd`; KLW/DDM-consistent is
  `posterior_mean_sd`. Fixed behind `consistent_choice_noise` on branch
  `choice-noise-normalization`. (It shrinks the effects slightly but does **not**
  resolve the perceptual/memory split.)
- **Single-stuck-chain recovery** (`trim_stuck_chains`, notes §4b): every "soft"
  DDM cell was one stray chain (often *high*-lp, a sharp minor mode — do not gate
  on lp); trimming recovers r̂≈1.00, ESS in the thousands. The DDM **is**
  recoverable.

## The one path left to separate perceptual vs memory (future work)

bauer's DDM/RDM put **both** encoding noises into the drift SNR symmetrically,
with no across-trial drift variability — so they *cannot* distinguish:
- **perceptual = within-trial** noise on the still-visible n2 → *reducible* by
  accumulation → costs RT, not asymptotic accuracy;
- **memory = across-trial** offset on the held n1 (a fixed degraded trace) →
  *irreducible* → acts as **drift variability (`sv`)** → hits accuracy and
  **produces slow errors**.

These have **distinct RT-distribution signatures** (across-trial `sv` → slow
errors), and we already observed the current model **underfits the slow-error
QPP wing** — the unmodelled signature. A model with **memory→`sv`,
perceptual→within-trial** is therefore the principled route to identifiability,
motivated by an observed misfit. It requires a new (drift-variability)
likelihood in bauer. Design lever for a future experiment: manipulate the n1→n2
**delay** over a wider range (memory should grow with delay; perceptual not).
