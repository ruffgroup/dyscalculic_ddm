# Figure captions

## Supplementary Figure X — `supp_ddm_vs_rdm.pdf`

**Both models reproduce the chronometric (difficulty) effect on response
time; only the racing-diffusion model also reproduces the overall-magnitude
effect.** Posterior-predictive summaries pooled across groups (33 control,
33 dyscalculic); open circles: empirical data; grey and teal lines:
posterior-predictive median RT under the fitted DDM and RDM, respectively
(shaded bands: 94% highest-density interval). (a) Median RT as a function
of discrimination difficulty, |log(n2/n1)|. Both the DDM and the RDM
capture the standard chronometric slope — harder (smaller-difference)
trials are slower. (b) Median RT as a function of overall stimulus
magnitude (n1 + n2), at fixed discriminability. The RDM reproduces the
characteristic decrease in RT with increasing overall magnitude; the DDM
instead predicts a roughly flat-to-rising RT, because its drift depends
only on the evidence difference between the two options, not their sum.
(c) Leave-one-out cross-validated model comparison (PSIS-LOO), expressed
as the difference in expected log pointwise predictive density (Δelpd_loo)
relative to the RDM (dotted line: 0; error bars: ±1 SE). The DDM scores
738 points lower — a decisive difference in the RDM's favour — though
Pareto-k diagnostics flagged 87 of 11,665 trials as unreliable (k > 0.7)
for the DDM fit.

## Supplementary Figure Y — `supp_rdm_params.pdf`

**Full posterior parameter summary for the trusted model — the RDM with
w0/ws freed ("RDM, full drift"), by group.** Row 1: population-level group
mean ± 94% HDI per parameter, in natural units (control blue, dyscalc
red) — the numbers quoted in `CONCLUSIONS.md`. Row 2: the dyscalc − control
contrast itself, ± 94% HDI, on the model's link/unconstrained scale (dotted
line: 0) — the actual posterior quantity the P(Δ>0) hypothesis test is
computed from; it is not a naive subtraction of the row-1 natural-unit
intervals, which would be invalid given the nonlinear (softplus) link and
the correlated group posteriors. Row 3: between-subject SD (τ) per group,
same link scale — whether dyscalculics are also more *heterogeneous*, not
just shifted on average. Bold P-values mark parameters whose 94% HDI
excludes 0. All x-axes are anchored at 0. Dyscalculics show credibly higher
perceptual noise, memory noise, threshold, and baseline drift/urgency
(all P≥0.99); non-decision time and the magnitude-sum drift ws are not
credibly different. Between-subject heterogeneity is comparable across
groups for every parameter except memory noise, where dyscalculics are
if anything *more* homogeneous (P=0.03).

## Supplementary Figure Z — `supp_subject_params.pdf`

**Individual-subject posterior estimates for the trusted model (RDM, full
drift), rank-ordered.** Each point: one subject's posterior mean ± 94% HDI
(natural units; control blue, dyscalc red), ranked within each panel by
point estimate, both groups pooled onto one x-axis. Dotted horizontal
lines and the diamonds past the vertical divider: population mean ± 94%
HDI per group (same values as `supp_rdm_params.pdf` row 1) — shown on the
same axis as the individual estimates so within-group spread and the
pooled estimate read together, the standard forest-plot "summary diamond"
convention.

## Supplementary Figure — `supp_recover_cross_pcm.pdf`

**A choice-only model reverses the noise difference it was simulated
from.** Parametric-bootstrap test of the pure PCM's paradoxical contrast
(dyscalculics *less* noisy, the opposite of the RDM's dyscalculics-noisier
result, P = 1.00). If dyscalculics offset extra noise with a higher
decision threshold, a model with no threshold parameter should misread the
resulting near-normal accuracy as lower noise. Each of 1,000 replicates
draws one plausible truth from the RDM (full drift) posterior, in which
perceptual noise and threshold are both genuinely higher in dyscalculics,
simulates choices and response times for the full 66-subject design,
discards the response times, and refits the choice-only PCM to the
simulated choices. 997 replicates met the convergence criterion
(r̂ ≤ 1.05). (**a**) Grey violins: distribution of the true simulated
contrast (left; positive in 997/997 replicates) and of the refit PCM
contrast (right). Thin lines connect 150 randomly chosen replicates, red
where the sign flips. (**b**) All 997 refit contrasts. Grey line, mean
−0.15 (94% CI [−0.16, −0.14]); red dashed line, the real-data PCM estimate
(−0.19). Contrasts are dyscalculic − control on the model's unconstrained
(pre-softplus) scale; dotted line, 0. The refit reverses sign in 821/997
replicates (82%, 94% CI [80, 85]) although the generative noise difference
always favoured "dyscalculics noisier", and the real-data estimate falls
well inside the simulated distribution (36% of replicates were at least as
extreme). The PCM's paradox is thus expected under the RDM account, and
does not indicate a genuine perceptual advantage in dyscalculics. The
reversal is specific to the parameter the threshold trades off against:
the recovered *memory*-noise contrast stays positive (dyscalculic >
control) in 99.5% of replicates.

## Supplementary Figure — `supp_threshold_leak.pdf`

**The threshold difference is what the choice-only model reads as a
perceptual-noise difference.** Interventional version of the recovery test
in `supp_recover_cross_pcm.pdf`. There, every simulated world had both a
higher dyscalculic threshold and higher dyscalculic noise, so the test
could not say which of the two the PCM was misreading. Here one group
contrast at a time is rescaled on the model's unconstrained scale in the
generating posterior draw, leaving everything else in that draw untouched,
before choices and response times are simulated, the response times
discarded, and the choice-only PCM refitted to the choices. 40 replicates
per condition, of which the 35 that converged in all seven conditions
(r̂ ≤ 1.05) are shown. All error bars are bootstrap 94% CIs of the mean
over replicates, matching the 94% posterior intervals used elsewhere in
this supplement. (**a**) Dose-response. The simulated threshold contrast is
set to 0, 0.5, 1 and 1.5 times its fitted value (x axis; mean dyscalculic
threshold in natural units in brackets, control 0.50), while the true
perceptual-noise contrast is held at its fitted value throughout (grey
line, +0.33, 94% CI [+0.33, +0.34]). Grey lines, individual replicates;
teal, mean and 94% CI. What the PCM returns tracks the threshold almost one
for one (slope −1.00, 94% CI [−1.09, −0.92]; negative in 35/35 replicates)
and crosses zero close to the fitted threshold difference. With the
threshold contrast removed the PCM recovers the correct sign (+0.21, 94% CI
[+0.16, +0.25]); at 1.5 times the fitted threshold contrast it returns
−0.36 (94% CI [−0.44, −0.28]). (**b**) Which contrast leaks. Filled
circles, contrast recovered by the PCM (teal, perceptual noise; gold,
memory noise); vertical ticks, the true simulated value; red dashed line,
the real-data PCM estimate (−0.19); dotted line, 0. Removing the true noise
contrasts while keeping the threshold difference still yields −0.19 (94% CI
[−0.25, −0.14]), reproducing the real-data estimate in a world with no
perceptual-noise difference at all. Removing the urgency contrast instead
barely moves the estimate (−0.18, 94% CI [−0.24, −0.13]), and with every
group contrast at 0 the recovered perceptual-noise contrast is 0 within
error (+0.04, 94% CI [−0.01, +0.09]), so the refit is unbiased for this
parameter. The threshold also leaks into recovered *memory* noise, in the
opposite direction and three times more weakly (slope +0.34, 94% CI [+0.28,
+0.40]), and memory noise sits about 0.23 below its true value in every
condition, including the all-zero control (−0.23, 94% CI [−0.34, −0.13]).
That offset belongs to the choice-only refit rather than to the
intervention, and it is small in context: single-fit 94% HDIs are about 1.1
wide on this scale against 0.5 for perceptual noise, only 6 of 35 all-zero
fits excluded 0 individually, and 0.23 here is roughly 0.01 in natural SD
units, since memory noise is near 0.05 and the softplus link is close to
exponential in that range. (**c**) What the PCM sees. Mean simulated
accuracy per group. As fitted, the higher threshold nearly cancels the
accuracy cost of the extra noise (86.6% vs 84.2%, with dyscalculic median
RT 1.25 s against 1.09 s). With the threshold contrast at 0 the accuracy
gap doubles (86.6% vs 81.4%) and dyscalculics become the faster group
(1.00 s). With only the threshold differing, they are the more accurate
group (88.2%). A model with no threshold parameter can express these
patterns only as differences in noise.
