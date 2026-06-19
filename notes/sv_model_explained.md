# The sv-DDM, explained: how noise in the first number drives shrinkage

This note explains, from the ground up, (1) how Bayesian shrinkage works in the
magnitude-comparison observer, (2) why the **noise on the first number (n1)**
is what creates the compression effect, and (3) how the `memory_as_sv` model
turns that same noise into a *temporal* signature (slow errors) that lets us
separate perceptual from memory noise. The terse derivation lives in
`~/git/bauer/notes/memory_as_sv.md`; this note is the intuition.

## 1. The task and the observer

On each trial the subject sees **n1**, then **n2**, and decides which is
larger. The observer encodes each number on a **log scale** with Gaussian
noise, and holds a **prior** over log-magnitude `V ~ N(μ_p, σ_p²)` (learned from
the range of numbers in the experiment).

So for each stimulus k ∈ {1, 2} the observer gets a noisy sensory sample

```
x_k = V_k + ε_k,     ε_k ~ N(0, σ_e,k²)
```

and must form a best estimate of the true log-magnitude `V_k`.

## 2. Shrinkage: the noisier the estimate, the harder it is pulled to the prior

The Bayes-optimal estimate combines the noisy sample `x_k` with the prior. For
the linear-Gaussian case it is a **precision-weighted average**:

```
μ̂_k = w_k · x_k  +  (1 − w_k) · μ_p

           σ_p²                 reliability of the sample
w_k = ───────────────   =   ───────────────────────────────────
        σ_p² + σ_e,k²         reliability of sample + reliability of prior
```

`w_k` is the **reliability weight** (a number between 0 and 1):

- If the sample is **precise** (`σ_e,k` small), `w_k → 1`: trust the sample,
  estimate ≈ `x_k`, almost no shrinkage.
- If the sample is **noisy** (`σ_e,k` large), `w_k → 0`: distrust the sample,
  estimate ≈ `μ_p`, the estimate is **pulled strongly toward the prior mean**.

This pull-toward-the-prior is **shrinkage** (a.k.a. regression to the mean). The
key point for everything below:

> **More encoding noise on a stimulus → smaller `w_k` → its estimate is
> shrunk *more* toward the prior mean.**

## 3. Why n1 is the one that drives compression

In this task the two numbers are **not encoded equally well**:

- **n2** is still on the screen (or just was) when the decision is made — it
  carries only **perceptual** noise: `σ_e,2 = σ_perc`.
- **n1** was shown earlier and must be **held in memory** while n2 is encoded —
  it carries perceptual noise **plus an extra memory corruption**:
  `σ_e,1 = σ_perc + σ_mem`  (n1 is strictly noisier than n2).

Because `σ_e,1 > σ_e,2`, the weights are ordered:

```
w₁ < w₂          (the first number is the less reliable one)
```

So **n1 is shrunk toward the prior more than n2 is.** That asymmetry is the
entire compression effect:

- When both numbers are **small** (below the prior mean), shrinkage pushes both
  estimates *up* toward `μ_p`, but it pushes **n1 up more** than n2. The
  observer's internal n̂1 is inflated relative to n̂2 → it more often judges
  **n1 ≥ n2** → it under-reports "chose n2".
- When both numbers are **large** (above the prior mean), shrinkage pulls both
  *down*, but **n1 down more** → n̂1 deflated relative to n̂2 → it more often
  judges **n2 ≥ n1** → it over-reports "chose n2".

Net result: **P(chose n2) rises with overall magnitude** — the compression /
regression-to-the-prior curve we plot against n1. It exists **only because
n1 is noisier than n2** (if `σ_mem = 0`, then `w₁ = w₂`, both shrink equally,
and the curve is flat). This is why **memory noise (the n1-specific part) is the
direct driver of compression**, while perceptual noise — common to both — mostly
just lowers overall accuracy.

> Sharpen it: the size of the compression is governed by the **reliability gap**
> `w₂ − w₁`. Adding **memory** noise raises only `σ_e,1`, opening the gap →
> *more* compression. Adding **perceptual** noise raises both `σ_e,1` and
> `σ_e,2`, so at matched accuracy it opens the gap much less → *less*
> compression. That difference is exactly what the forward-simulation figure
> (`notes/figures/sim_perc_vs_mem.pdf`, top row) shows, and it is what makes the
> two noises distinguishable from choices in principle.

## 4. From a static estimate to a decision over time (the DDM)

The DDM doesn't just compare two point estimates; it **accumulates evidence**
for "n2 > n1" over time until it hits a boundary, which also yields a reaction
time. The drift rate (speed/direction of accumulation) is the **signal-to-noise
ratio** of the estimate difference:

```
        μ̂₂ − μ̂₁              (how far apart the two shrunken estimates are)
v  =  ─────────────
          sd_DV               (the noise the accumulator fights against)
```

The standard bauer DDM puts the *combined* encoding noise of both stimuli into
`sd_DV`, so perceptual and memory noise enter `v` **symmetrically** — the data
can only constrain their **sum**. That is the source of the
perceptual-vs-memory trade-off in the standard DDM: from RT alone, raising
either noise lowers the drift the same way.

## 5. The `memory_as_sv` idea: give the two noises different jobs in time

The fix is physical, not statistical. The two noise sources differ in **when**
they can be reduced:

- **Perceptual noise is *reducible*.** n2 is available; the accumulator can keep
  taking fresh looks. So perceptual noise is **within-trial diffusion noise** —
  it slows you down but you average it away. → it belongs in `sd_DV`.
- **Memory noise is *frozen*.** The degraded memory trace of n1 is fixed for the
  whole trial — you cannot re-sample a better memory mid-decision. So on each
  trial it acts as a **constant offset to the drift rate**, which varies *from
  trial to trial*. That is exactly **Ratcliff across-trial drift variability,
  `sv`**.

Crucially, **shrinkage is unchanged** by this re-interpretation. The drift mean
`v` still uses the shrunken estimates `μ̂₁, μ̂₂` with the same gains `w₁, w₂`,
so n1 is still shrunk more, and the **compression curve is preserved** (the
sv-DDM reproduces the choice proportions — see the simulation figure, panel c).
What changes is that the *frozen part* of n1's noise now lives in `sv` instead
of in `sd_DV`:

```
within-trial noise   sd_DV = sqrt( (w₂·σ_perc)² + (w₁·σ_perc)² )   ← perceptual only
drift mean           v     = v_scale · (μ̂₂ − μ̂₁ + threshold) / sd_DV
drift variability    sv    = v_scale · (w₁·σ_mem) / sd_DV          ← frozen memory of n1
```

Read the `sv` line in words: the **memory noise on n1** (`σ_mem`), seen through
n1's **shrinkage gain** (`w₁`), and scaled by the same SNR normaliser as the
drift, *is* the trial-to-trial wobble in the drift rate.

## 6. Why this finally separates perceptual from memory

The two noises now leave **two independent fingerprints**:

| Signature | Perceptual noise | Memory noise |
|---|---|---|
| Overall accuracy | ↓ (strong) | ↓ |
| **Compression** (choice) | weak (symmetric shrinkage) | **strong** (n1-only shrinkage) |
| **Slow errors** (RT tail) | none | **yes** (`sv` → trials with a bad n1 memory drift the wrong way slowly) |

The static choice model already has the **compression** handle (§3). The sv-DDM
adds the **slow-error** handle on top. With two orthogonal signatures for two
parameters, perceptual and memory become **separately identifiable** — confirmed
by parameter recovery (clean double dissociation; see `validate_sv.py` and
`notes/CONCLUSIONS.md`).

## 7. The intuition in one paragraph

The first number is remembered, not seen, so it is encoded more noisily than the
second. Bayes pulls the noisier estimate harder toward the average magnitude, so
n1 is shrunk more than n2 — and that imbalance *is* the compression effect: small
pairs look "n1-ish", large pairs look "n2-ish". Memory noise is the part of n1's
noise that is frozen for the trial, so on a bad-memory trial the whole decision
drifts the wrong way slowly: memory therefore shows up twice — as **stronger
compression in the choices** and as **slow errors in the RTs** — whereas
perceptual noise, shared by both numbers and re-sampleable, mostly just lowers
accuracy. Those distinct fingerprints are what let us tell the two apart.
