# Why the perceptual/memory noise attribution flips between the static-choice
# model and the sequential-sampling models

## 1. Crisp answer

All three model classes (static **choice**, **DDM**, **RDM**) **agree** that
dyscalculics have a noisier *first* stimulus — the observable n1 encoding SD
`σ_n1 = σ_perceptual + σ_memory` is credibly larger in dyscalculics in **every**
fit (Δσ_n1 > 0 with P = 0.91 / 0.97 / 1.00 for choice / DDM / RDM). What flips is
**only the n2 (perceptual-only) component**: the static choice model pushes
`σ_n2 = σ_perceptual` *down* in dyscalculics (Δσ_n2 ≈ −0.05, P ≈ 0.02), whereas
the SSMs push it *up* (DDM Δσ_n2 ≈ +0.05, P ≈ 0.97; RDM ≈ +0.21, P ≈ 1.00). The
"perceptual−, memory+" verdict of the choice model and the "perceptual+,
memory≈0/+" of the SSMs are therefore two projections of the **same soft ridge**:
σ_perceptual and σ_memory enter `σ_n1` additively, so the choice likelihood —
which sees only accuracy + compression — cannot tell apart "raise n1's memory
term, lower the shared perceptual term" from "raise both," because both leave
accuracy and compression essentially unchanged while moving the split along the
ridge. The SSMs add **RT-distribution** information, and RT depends on the *drift
magnitude*, which is NOT constant along that ridge: at matched choice behaviour
the perc-down/mem-up corner predicts essentially no RT slowing, while dyscalculics
*are* slower. RT therefore rotates the estimate onto the both-noises-up corner,
which has a credibly raised perceptual term. The flip is **mostly benign ridge
movement** (the models concur on σ_n1, the better-identified quantity), with a
**small substantive layer**: the SSMs additionally claim that the *still-visible*
n2 is also noisier in dyscalculics, a claim the choice model cannot make because
it carries no RT signal to support it and instead spends that degree of freedom
compensating along the ridge with the opposite sign.

---

## 2. The mechanism, worked through

### 2.1 The shared front-end and what each likelihood reads off it

All models share the Bayesian observer (`bauer/models/magnitude.py`,
`shared_perceptual_noise`):

```
σ_e,1 = σ_perceptual + σ_memory        (n1, held in memory)   [magnitude.py:106]
σ_e,2 = σ_perceptual                    (n2, still visible)     [magnitude.py:107]
w_k   = σ_p² / (σ_p² + σ_e,k²)          reliability weight
μ̂_k  = w_k·x_k + (1−w_k)·μ_p           posterior mean (shrinkage to prior)
```

with the empirical prior `μ_p = 2.263, σ_p = 0.596` (log-space; computed from the
data). The two encoding SDs add **in SD space** to make `σ_e,1`, so the only thing
that distinguishes a perceptual from a memory unit is that perceptual loads on
*both* stimuli and memory loads on *n1 only*. The decision-relevant difference
signal `μ̂₂ − μ̂₁` is common to all three models; the models differ purely in the
response layer:

* **Choice** (`core.py:134-165`): `P(chose n2) = Φ( (μ̂₂−μ̂₁) / √(sd₁²+sd₂²) )`,
  where `sd_k` is the raw `σ_e,k` (default) or the posterior-mean SD `w_k·σ_e,k`
  (the KLW / `consistent_choice_noise` variant). **No free scale**, no RT, no
  threshold. Each trial contributes a single Bernoulli probability.
* **DDM** (`ddm.py:567-611`): drift `v = (μ̂₂−μ̂₁) / √((w₁σ₁)² + (w₂σ₂)²)`, with a
  **free boundary `a`** and non-decision time `t0`; `v_scale = 1` fixed. The full
  RT distribution per response enters the Wiener first-passage likelihood.
* **RDM** (`race.py:409-480`): advantage-race drifts `v_{1,2} = w₀ ± w_d·Δμ̃ +
  w_s·Σμ̃` with within-trial noise `σ_k = w_k·σ_e,k`, common boundary `a`, `t0`.
  Free drift-scale knobs (w₀, w_d, w_s) plus `a`, `t0`.

So the **choice model has exactly two behavioural observables to constrain two
noise parameters**: (i) overall accuracy / psychometric slope, and (ii) the
compression / n1-dependence (regression-to-prior). The SSMs see those *plus* the
RT distribution and its split by correct/error.

### 2.2 The ridge: what the choice likelihood can and cannot separate

Compression is the regression-to-the-prior signature and is governed by the
**reliability gap** `w₂ − w₁` (the held n1 is shrunk toward the prior more than
the visible n2, biasing the choice toward n2 as n1 grows). In this regime
(σ_e ≈ 0.1–0.3 ≪ σ_p ≈ 0.6) the gap is driven by the n1/n2 SD *asymmetry*, i.e.
by `σ_memory`. Accuracy is driven mainly by the total discriminative noise
`√(σ_e,1² + σ_e,2²)`. The numerical experiment (run on the actual stimuli; see
§4.3) makes the geometry explicit — holding `σ_n1 = σ_perc + σ_mem` fixed at 0.30
and trading perc→mem:

| σ_perc | σ_mem | σ_n1 | σ_n2 | P(correct) | w₂−w₁ (compression) | mean drift |
|-------:|------:|-----:|-----:|-----------:|--------------------:|-----------:|
| 0.20 | 0.10 | 0.30 | 0.20 | 0.787 | 0.101 | 1.13 |
| 0.15 | 0.15 | 0.30 | 0.15 | 0.808 | 0.143 | 1.28 |
| 0.10 | 0.20 | 0.30 | 0.10 | 0.823 | 0.175 | 1.43 |
| 0.05 | 0.25 | 0.30 | 0.05 | 0.832 | 0.195 | 1.55 |
| 0.02 | 0.28 | 0.30 | 0.02 | 0.835 | 0.201 | 1.59 |

Across this trade, accuracy moves by only ~0.05 while compression nearly doubles.
**Two summary observables, two parameters, along a near-flat valley** — exactly the
posterior collinearity reported on the real data (r ≈ −0.81). The likelihood pins
`σ_n1` (the better-identified sum, which sets the discriminative noise *and*
anchors the compression) well, but the *split* of `σ_n1` between perceptual and
memory is weakly identified.

### 2.3 Why the choice model lands on perceptual−, memory+

To explain the dyscalculic data the choice model must increase **compression**
(dyscalc show a steeper n1-dependence) without crashing accuracy too much. The
fastest way along the ridge to buy more compression is to make the n1/n2 SDs
*more* asymmetric: push the memory term (n1-only) **up** and the shared perceptual
term **down**. In the table, moving from (0.10, 0.10)-like control toward
(0.05, 0.20)-like dyscalc raises compression from ~0.074 to ~0.143 while *keeping
accuracy ~0.86* — a near-perfect fit to "more compression, slightly lower
accuracy." That corner has Δσ_perceptual < 0 and Δσ_memory > 0. This is precisely
the observed choice verdict (Δσ_perc ≈ −0.047, P ≈ 0.02; Δσ_mem ≈ +0.068,
P ≈ 1.00; identical for raw and KLW, see §4.1). The negative perceptual sign is a
**ridge-compensation artifact**, not a claim that dyscalculics see *better*: σ_n1
still goes up (+0.021); the model is simply re-attributing within the sum.

### 2.4 Why RT rotates the SSM onto perceptual+

The DDM/RDM drift magnitude is the **decision-relevant SNR**, and it is *not*
constant along the choice ridge: the last column of the table shows mean drift
climbing 1.13 → 1.59 as we move toward the perc-down corner. Lower perceptual
noise ⇒ less within-trial diffusion noise ⇒ **higher drift ⇒ faster RT**. So the
two ridge corners that the choice model treats as nearly equivalent make
**opposite RT predictions**:

* perc-down / mem-up (choice corner): drift ≈ control ⇒ **predicts ~no slowing**.
  Matched-behaviour check: control (0.10,0.10) drift 1.78 → dyscalc-as-choice
  (0.05,0.20) drift 1.72 (−3%).
* both-up (SSM corner): drift **drops** ⇒ predicts slowing. dyscalc-as-SSM
  (0.15,0.15) drift 1.28 (−28%).

The raw data say dyscalculics are **slower** (median RT +0.15 s; correct-RT
control 1.04 s vs dyscalc 1.19 s) **and slightly less accurate** (0.875 vs
0.854). The only way an accumulator with a near-static boundary reproduces *both*
slower RT and lower accuracy is to **lower the drift**, i.e. raise the *common*
(perceptual) noise. Hence the SSMs rotate off the choice corner onto the
both-noises-up corner, where the perceptual contrast is positive. RT is the extra
constraint that "uses up" the ridge degree of freedom the choice model left free.

### 2.5 The role of the free boundary / drift scale — and why it is *not* the
driver here

A natural worry is that the SSMs absorb the group difference into the boundary
`a` or the drift scale (`w₀…`) and thereby free the noise terms to move the other
way. The traces rule this out: the group effect on `a` is small and not credible
(DDM Δa = +0.057, P = 0.90 in the shared fit; RDM Δa = +0.009, P = 0.59).
Even in the DDM variant that explicitly frees non-decision time to soak up the
slowing (`ddm_t0group`), t0 takes only part of it (Δt0 = +0.055, P = 0.92) and the
perceptual contrast **stays positive** (+0.029, P = 0.73), memory null. So the SSM
verdict (perceptual+) is *robust to* giving the model alternative RT knobs — the
slowing is genuinely attributed to lower drift = more common noise, not to caution
or motor time. The free boundary matters in a different way: because `a` sets the
choice scale, the SSM does not need the noise terms to also carry the overall
accuracy level the way the choice model's single `√(sd₁²+sd₂²)` normalizer does.
This decouples accuracy from the noise split and lets RT (drift) dominate the
split — which is exactly what rotates it.

### 2.6 Why memory is + for choice but ≈0 for DDM specifically

In the choice model the *only* lever for the extra dyscalculic compression is the
n1/n2 asymmetry, i.e. the memory term, so compression → memory↑ (P ≈ 1.00). The
DDM has additional levers for the same phenomena: (i) drift dynamics and the free
boundary can reshape the accuracy/compression mapping, and (ii) once RT forces the
*common* perceptual term up, σ_n1 = σ_perc + σ_mem is already raised enough to
match the data, so the residual memory contrast collapses to null (DDM Δσ_mem
≈ +0.005, P ≈ 0.62). The RDM sits in between (Δσ_mem ≈ +0.023, P ≈ 0.90): its
advantage-drift `w_d·Δμ̃` term re-expresses part of the n1/n2 asymmetry through the
drift structure, but it still retains a weak positive memory contrast on top of a
strong perceptual one.

---

## 3. (a) benign ridge vs (b) substantive misspecification — verdict

**Hypothesis (a):** both likelihoods sit on the same σ_perc↔σ_mem ridge; they
project onto different points, but the *joint* / observable quantity (the
per-stimulus SDs σ_n1, σ_n2) agrees.

**Hypothesis (b):** the dyscalculic deficit is genuinely in something the choice
model lacks (an RT-side quantity), and the choice model, unable to represent it,
launders it into the noise split with a misleading sign.

**The decisive test** is the cross-model comparison on the *observable*
per-stimulus SDs from the **n1/n2-separate** parameterization (the cleanest common
ground — no perc/mem reparam, both noises free per stimulus). Computed directly
from the converged traces (§4.2):

| model | Δσ_n1 (=perc+mem) | Δσ_n2 (=perc) |
|-------|------------------:|--------------:|
| choice (raw) | +0.029 (P 0.96) | **−0.050 (P 0.02)** |
| choice (KLW) | +0.028 (P 0.96) | **−0.051 (P 0.02)** |
| DDM | +0.065 (P 1.00) | +0.020 (P 0.71) |
| RDM | +0.256 (P 1.00) | +0.212 (P 1.00) |

**Verdict: predominantly (a), with a thin (b) layer concentrated entirely in σ_n2.**

* **On σ_n1 all four models agree (positive, P ≥ 0.96).** This is the
  better-identified quantity (it sets discriminative noise and anchors
  compression), and the choice/DDM/RDM concur on it. The headline "dyscalculics
  have a noisier number representation" is therefore **model-robust** — exactly
  what we want from (a).
* **The entire disagreement is on σ_n2** (the perceptual-only, still-visible
  stimulus). Here choice says *negative* and the SSMs say *zero-to-positive*.
  This residual is the (b) component: the choice model, blind to RT, **cannot**
  see that n2 is also degraded, and worse, it is *forced* to set Δσ_n2 negative as
  the ridge-compensation for raising the n1-asymmetry needed to fit compression.

The negative σ_n2 in the choice model is the red flag flagged in the brief
("accuracy bookkeeping"): dyscalculics are **less** accurate, yet the choice model
assigns them **less** perceptual noise. That is internally consistent only because
the memory term simultaneously absorbs all the accuracy/compression work; the
perceptual term is then pulled the opposite way along the anti-correlated ridge.
The SSMs, with RT in hand, resolve this by raising both — the more physically
sensible reading. So the flip is *mostly* a relabeling of the same σ_n1 increase
(benign), but the choice model's specific *sign* on the perceptual component is a
genuine artifact of its missing RT channel (substantive, but localized to the n2
term).

---

## 4. Concrete numbers

All from the converged long chains in `results/` (4 chains × 4000 draws). The
group contrast is treatment-coded; the observable group Δ on a softplus-link
parameter is `softplus(β₀+β₁) − softplus(β₀)`, applied to the linear-predictor
coefficients `*_mu` (the softplus is applied to the *linear predictor*, not
per-coefficient — `core.py:get_trialwise_variable`, lines 851-872, confirmed).

### 4.1 Shared (perceptual/memory) parameterization — reproduces the headline flip

| model | Δσ_perceptual | Δσ_memory |
|-------|--------------:|----------:|
| choice (raw) `model-choice_group_long_trace.nc` | −0.047 (P 0.02) | +0.068 (P 1.00) |
| choice (KLW) `model-choice_group_klw_trace.nc` | −0.046 (P 0.02) | +0.067 (P 1.00) |
| DDM `model-ddm_group_long_trace.nc` | +0.047 (P 0.97) | +0.005 (P 0.62) |
| RDM `model-rdm_group_long_trace.nc` | +0.210 (P 1.00) | +0.023 (P 0.90) |

Raw and KLW choice are essentially identical, confirming the flip is **not** the
decision-noise-normalization issue (`notes/choice_vs_ddm_decision_noise_inconsistency.md`).

### 4.2 n1/n2-separate parameterization — the cross-model common ground (table in §3)

The shared-fit and n1n2-fit numbers agree (e.g. DDM observable Δσ_n1: +0.052 from
the shared fit vs +0.065 from the n1n2 fit; choice Δσ_n2: −0.047 vs −0.050),
confirming the two parameterizations describe the same posterior.

### 4.3 Accumulator-scale group effects (shared fits)

* DDM Δa = +0.057 (P 0.90), small; RDM Δa = +0.009 (P 0.59), null.
* `ddm_t0group` fit: Δt0 = +0.055 (P 0.92), Δσ_perc = +0.029 (P 0.73), Δσ_mem
  ≈ 0 (P 0.46), Δa = +0.024 (P 0.65). RT slowing splits between t0 and lower
  drift; perceptual stays positive even with t0 free.

### 4.4 Raw-data bookkeeping (rt ≥ 0.20 s filter, as fit; 11 665 trials)

* Accuracy: control 0.874, dyscalc 0.854 (dyscalc **less** accurate).
* Median RT: control 1.07 s, dyscalc 1.22 s (dyscalc **+0.15 s slower**).
* Median RT by correctness: control 1.04 (correct) / 1.35 (error); dyscalc 1.19 /
  1.57. Errors are **slower** than corrects in both groups — the classic
  slow-error signature, more pronounced in dyscalc.

The combination "slower **and** less accurate" is the diagnostic pattern: it is
incompatible with a pure boundary or pure-perceptual-improvement account and forces
the SSM toward **lower drift = higher common (perceptual) noise**, against the
choice model's perc-down corner.

### 4.5 Forward-simulation ridge demonstration (§2.2 table)

Run on the actual stimulus set with the empirical prior. Shows accuracy nearly
flat (0.79→0.83) and compression near-doubling (0.10→0.20) along the σ_perc↔σ_mem
ridge at fixed σ_n1, while mean drift varies 1.13→1.59 — i.e. choice is degenerate
along the ridge but RT is not.

---

## 5. Caveats and falsifiable predictions

### Caveats
1. **The split is weakly identified by design.** ISI does not modulate the effect
   in this paradigm, so the one experimental handle that would physically separate
   memory (delay-dependent) from perceptual (delay-independent) noise is absent.
   The SSM "perceptual+" should be read as "the common encoding noise is up," not
   as a strong dissociation from memory.
2. **bauer's DDM/RDM put both encoding noises into the drift SNR symmetrically**
   (`_drift_from_snr`, `_drifts_from_post_and_prior`) with no across-trial drift
   variability in the standard variants. So within those likelihoods the perc/mem
   split is *still* on a (shallower) ridge — RT only partially lifts the
   degeneracy. The `memory_as_sv` variant (`_drift_and_sv_from_snr`, ddm.py:614)
   is the principled fix: it routes memory → Ratcliff `sv` (frozen n1 trace →
   across-trial drift variability → slow errors) and perceptual → within-trial
   diffusion, giving the two components *distinct* RT-distribution signatures.
3. **RDM σ magnitudes are on a different scale** (σ_n1 ≈ 0.7–1.0 vs ≈ 0.2–0.3 for
   choice/DDM) because the race absorbs scale into w₀/w_d and the within-trial
   `σ = w·σ_e` floor; compare *signs and P-values*, not raw magnitudes, across
   architectures.

### Falsifiable predictions / suggested follow-ups
1. **`memory_as_sv` DDM should break the tie cleanly.** Prediction: if the
   dyscalculic deficit is genuinely common/perceptual, the sv-DDM gives
   Δσ_perceptual > 0 and Δsv ≈ 0; if it is memory, Δsv > 0 (more slow errors) and
   Δσ_perceptual ≈ 0. The current standard DDM under-fits the slow-error QPP wing
   (noted in `CONCLUSIONS.md`), so this model is motivated by an observed misfit —
   fit `ddm_sv_group` and check the slow-error tail and the Δσ_perc vs Δsv split.
2. **Within-model SSM recovery (the recover_ponly / recover_monly cluster jobs).**
   Prediction: the DDM recovers perceptual-truth→perceptual and
   memory-truth→memory with the correct sign (unlike the choice model, which
   returns memory↑ regardless of the generating truth — already shown in
   `recover_choice_*.csv`). If the DDM also fails to recover the sign, the SSM
   perceptual+ is itself ridge noise and (b) shrinks toward zero.
3. **A delay manipulation** (vary n1→n2 ISI over a wide range) is the design-side
   test: memory noise should grow with delay, perceptual should not. Prediction:
   Δσ_n1 (but not Δσ_n2) should increase with ISI in dyscalculics; if neither
   moves, the deficit is purely perceptual/common and the "memory" label is
   vacuous.
4. **PPC discriminator:** simulate from the choice-corner (perc−/mem+) vs
   SSM-corner (both+) parameter sets and compare predicted RT for dyscalculics.
   The choice corner predicts ~no slowing (drift ≈ control); the data show +0.15 s.
   A PPC of mean RT by group under the choice-implied noise split should
   *underpredict* the dyscalculic slowing — a direct falsification of the choice
   model's perceptual-negative attribution.
