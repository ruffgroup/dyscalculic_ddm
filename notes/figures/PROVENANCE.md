# Provenance — `notes/figures/`

What generated each file here, from what data, and (for the one that had to
be reconstructed) exactly what's exact versus approximated. Written after a
2026-08-18 incident where a figure's plotting script (behind
`supp_ddm_vs_rdm.{pdf,png}`) turned out to have never been committed and
couldn't be found anywhere on disk — see that entry below for the full
story. This file exists so it doesn't happen silently again: any script
that produced something in this directory should be listed here the moment
it's written, not after the fact.

## `supp_threshold_leak.{pdf,png}` (new, 2026-08-24)

`plot_supp_threshold_leak.py` (repo root) ← `notes/data/threshold_leak.tsv`
(committed, 204 KB; the 280 per-run rows aggregated) ←
`results/recover_threshold_leak/rep*_*.csv` (gitignored) ←
`recover_threshold_leak.py` (repo root), run on the cluster via
`run_recover_threshold_leak.sh` as SLURM array `5240965`, `--array=0-279%120`
(40 reps x 7 conditions, ~10 min each, CPU-only, lowprio) against
`/shares/zne.uzh/gdehol/dyscalculic_ddm/model-rdm_max_cont_hn_trace.nc`, with
bauer **v0.4.0** pinned the same way `run_fit.sh` does
(`PYTHONPATH=$HOME/bauer-030`). Also reads `notes/data/overview.json` for the
real-data PCM contrast. First-run analysis, nothing reconstructed. The plot
script prefers the per-rep CSVs when present locally and refreshes the TSV
from them, else reads the TSV, so the figure regenerates from git alone.

**What this answers, and why it needed new simulations.**
`supp_recover_cross_pcm` is observational: all 997 of its replicates had BOTH
a higher dyscalc threshold and higher dyscalc noise, so it cannot attribute
the PCM's sign reversal to either. Regressing the recovered contrast on the
per-draw true contrasts across those replicates is underpowered by
construction (the posterior's spread in the threshold contrast is SD 0.10
against a mean of +0.37 and never crosses 0, so there is no counterfactual in
the sample); that regression gives b = -0.18 per link-unit (p = 0.002) and
extrapolates to only +0.07 of the -0.15. The interventional runs recover the
same coefficient as -1.00 [-1.09, -0.92], i.e. the observational slope is
attenuated roughly 5x.

**Mechanics of the intervention** (the part to check if this is ever
re-run): in the cell-means parameterization the per-subject coefficient array
`{p}` has dims (chain, draw, subject, `{p}_regressors`) and lives on the
*link* (pre-softplus) scale, because bauer's regression path applies the
design matrix first and the transform after
(`RegressionMixin.get_trialwise_variable`, bauer v0.4.0 `bauer/core.py`).
Each subject's design row is one-hot for their group, so adding
`(scale - 1) * Delta` to the `C(group)[dyscalc]` column rescales exactly the
group contrast, preserves every subject's deviation from their own group
mean, and is a no-op for controls. `RaceMixin.ppc` simulates straight from
that array, so the intervention is exact rather than an approximate
re-parameterization. Conditions: `full`, `a_x0`, `a_x0.5`, `a_x1.5`,
`w0_x0`, `noise_x0`, `all_x0`. The last one is deliberately NOT named
`null`: `pandas.read_csv` reads that string back as NaN.

**Result.** 35/40 reps converged in all seven conditions (r̂ <= 1.05; 6 of
280 runs failed the threshold). All intervals below are bootstrap **94%**
CIs of the mean over reps, deliberately 94% and not 95% so the recovery
figures use the same interval mass as the posterior HDIs in the rest of the
supplement (`ci()` in `plot_supp_threshold_leak.py` sets HDI_PROB = 0.94,
and the `supp_recover_cross_pcm` caption was moved to 94% at the same time).
Recovered perceptual-noise contrast: +0.205 [+0.164, +0.247] at threshold
contrast 0, +0.051 at 0.5x, -0.118 at 1x, -0.355 [-0.435, -0.279] at 1.5x,
with the true contrast held at +0.333 throughout; per-rep slope -1.00
[-1.09, -0.92], negative in 35/35. With the noise contrasts zeroed but the
threshold kept, the PCM still returns -0.193 [-0.245, -0.140], essentially
the real-data -0.189. With every contrast zeroed it returns +0.038 [-0.011,
+0.088] (unbiased). Zeroing the urgency contrast instead moves it to -0.181
(slightly MORE negative than `full`), so the leak is specific to `a`. The
threshold leaks into recovered memory noise too, opposite in sign and three
times weaker (slope +0.336 [+0.278, +0.395]), and memory carries a further
downward offset of about 0.23 in every condition including the all-zero
control (-0.231 [-0.336, -0.130]); that one is a property of the PCM refit,
not of the intervention, and is worth only ~0.01 in natural SD units
(memory noise ~0.05, where softplus is near-exponential), against a
single-fit 94% HDI width of ~1.1.

## `supp_recover_cross_pcm.{pdf,png}` (new, 2026-08-18)

`plot_supp_recover_cross_pcm.py` (repo root, committed) ←
`notes/data/recover_cross_pcm.tsv` (committed, 199 KB — the 1000 per-rep
rows aggregated; `results/` itself is gitignored) ←
`results/recover_cross_pcm/rep*.csv` ← `recover_cross_pcm.py` (repo root, committed), run on the
cluster via `run_recover_cross_pcm.sh` as SLURM array `--array=0-999%150`
against `/shares/zne.uzh/gdehol/dyscalculic_ddm/model-rdm_max_cont_hn_trace.nc`
← `notes/data/overview.json`'s `choice` entry (the real committed PCM
contrast, for reference).

**What this answers.** Christian's comment on the pure-PCM result ("test
whether the response effects compensate for the noise") plus the
committed paradox: the choice-only PCM finds dyscalc noise LOWER than
control (ctrl 0.104 vs dys 0.087; reversed from `rdm_max`'s dys > ctrl,
P=1.00). A naive subject-level correlation between PCM noise and RDM
threshold/decision parameters is weak — but that's expected (shrunk
point-estimate correlations are attenuated, and the compensation account
doesn't require tight individual-level coupling, only that noise AND
threshold jointly determine what the PCM reads as "noise"). This script
tests the mechanism directly instead: simulate choices from the REAL
rdm_max posterior (so both the noise increase and the threshold increase
for dyscalc are genuinely baked in), drop the RTs, refit the pure PCM to
the simulated choices, and check whether the paradox reappears. It does:
first run at 24 reps (18/24, 75%) flipped sign; rerun at 1000 reps for a
tighter estimate — 997/1000 converged (r̂≤1.05), 24/24→997/997 had TRUE
dyscalc noise higher in every rep, but the refit PCM's contrast was
negative in 821/997 (82.3%, 95% CI [80.0%, 84.7%]), mean −0.152 (95% CI
[−0.164, −0.141]) — close to the real PCM's −0.189. Memory noise, run as
a built-in specificity check, almost never flips (1% of reps) — the
confound is specific to the parameter threshold trades off against, not
generic PCM refit noise. Runs faithfully replicate the production recipe
(bauer **v0.4.0** pinned via `PYTHONPATH=$HOME/bauer-030` in the SLURM
wrapper, matching `run_fit.sh` — NOT the `bauer_cuda` conda env's own
pip-installed bauer, which can be a different/newer checkout). Not
exact/reconstructed — this is a first-run analysis, no approximation
involved.

**2026-08-24 update (figure-ready pass + reproducibility fix).** The long
italic paragraph under the panels is gone — its content is now the caption
in `CAPTIONS.md`, where a journal expects it — and the figure title is one
line. In-panel labels moved into empty space and are in sentence case;
panel b carries the two key numbers (recovered mean, real-data PCM) as a
small keyed text block instead of two colliding line labels. Also: this
entry previously called the per-rep CSVs "committed" — they were not
(`results/` is gitignored). They now aggregate into
`notes/data/recover_cross_pcm.tsv`, and `plot_supp_recover_cross_pcm.py`
prefers the per-rep CSVs when the cluster output is present locally
(refreshing the TSV from them), falling back to the TSV otherwise, so the
figure regenerates from git alone. Verified by rebuilding.

## `supp_rdm_params.{pdf,png}`

`plot_supp_rdm_params.py` (repo root, committed) ← `notes/data/overview.json`'s
`rdm_max` entry (committed). No new data, no reconstruction — first-run script.

## `supp_subject_params.{pdf,png}`

`plot_supp_subject_params.py` (repo root, committed) ←
`notes/data/subject_params.tsv` (committed, small — 396 rows) ←
`collect_subject_params.py` (repo root, committed), run on the cluster
against `/shares/zne.uzh/gdehol/dyscalculic_ddm/model-rdm_max_cont_hn_trace.nc`
(per-subject `{param}` posteriors aren't in the committed `overview.json`
summary, only group-level ones — this script pulls them straight from the
trace) — plus `notes/data/overview.json`'s `rdm_max.raw` for the
population-mean reference diamonds. No reconstruction — first-run script.

## `overview_verdict.pdf`

`plot_overview.py` (repo root) ← `notes/data/overview.{json,npz}` ←
`collect_overview.py` ← the cluster posterior traces
(`/shares/zne.uzh/gdehol/dyscalculic_ddm/model-<label>_trace.nc`, 41 GB,
gitignored). Both `plot_overview.py` and `collect_overview.py` are
committed. Regenerate locally with `python plot_overview.py`; no traces
needed, `overview.{json,npz}` is committed and sufficient.

## `supp_ddm_vs_rdm.{pdf,png}`

`plot_supp_ddm_vs_rdm.py` (repo root, committed) ←
`notes/data/overview.{json,npz}` (committed, same source as above — no new
data needed).

**This is a reconstruction, not the original.** The original was generated
2026-08-17 11:26:06+02:00 (per the old PDF's embedded Matplotlib metadata)
by a script that was never committed and — after an exhaustive search
(`git log --all` incl. reflog/stash, full working tree incl. gitignored
paths, shell history, PDF/PNG metadata) — could not be recovered. It was
almost certainly run as an inline/ephemeral script by an earlier Claude Code
session and never saved to disk at all.

What's **exact** in the rebuild: panel B (LOO comparison) reads
`elpd_loo`/`elpd_se`/`bad_k` directly from `overview.json`'s `ddm`/`rdm`
entries — identical numbers to the original (ΔELPD-LOO = 738, matching the
old PNG's own printed "DDM: Δelpd -738" exactly). The Pareto-k annotation
is only shown when it's actually a caveat (DDM, 87/11,665 flagged); RDM's
own low bad-k count (7/11,665, well under the reliability threshold) isn't
called out in-panel — stating "reliable" everywhere adds no information
the absence of a flag doesn't already carry.

What's **reconstructed**: panel A (RT vs. n1+n2) pools the `control` and
`dyscalc` group curves already stored in `overview.npz`
(`{obsrt,ddm,rdm}__magnitude__{control,dyscalc}__{x,y|med,lo,hi}`) with a
simple equal-weight mean — valid because the design is exactly balanced
(33/33 subjects). This was **not guessed blind**: the pooled values were
computed and checked against a visual read of the original PNG (which was
still viewable before it got overwritten) before being trusted — data
points, DDM curve shape (rise then plateau ~1.20–1.25), and RDM curve shape
(rise then monotonic decline to ~1.09) all matched. The one thing that's a
genuine approximation rather than a verified match: the shaded HDI bands
average each group's own `lo`/`hi` bounds rather than recomputing a true
pooled-sample HDI (we only have per-group summary quantiles committed, not
the raw posterior-predictive draws) — visually close to the original but
not guaranteed numerically identical.

Model colors changed deliberately (grey DDM / teal RDM instead of the
original's grey/blue): blue already means "control" via `figstyle.GROUP`
everywhere else in this paper's figures, and this figure pools groups and
contrasts *models* instead, so reusing blue for "RDM" would read as a group
label by the color-persistence convention (`scientific-figures` skill).
Confirmed acceptable — the user only asked for the content to match, not
the exact look.

**2026-08-18 update:** added a third panel (a) plotting median RT vs.
discrimination difficulty (|log(n2/n1)|), same data source
(`obsrt__difficulty__*`, `{ddm,rdm}__difficulty__*` in `overview.npz` — not
new data, was already committed and unused). This strengthens the
figure's argument: it shows both models fit the standard chronometric
function equally well, so panel b's DDM failure on the magnitude effect
isn't just "the DDM fits RT badly" — it's specifically that the DDM's
drift depends only on the evidence difference, never the sum. Old panel A
(magnitude) is now panel B; old panel B (LOO) is now panel C. This panel
is **exact**, not reconstructed — no approximation involved.

## `supp_rdm_params.{pdf,png}` (new, 2026-08-18)

`plot_supp_rdm_params.py` (repo root, committed) ←
`notes/data/overview.json`'s `rdm_max` entry (committed — "RDM (full
drift)", the w0/ws-freed fit that CONCLUSIONS.md's headline numbers come
from). No new data; no reconstruction involved — this is a first-run
script, not a rebuild.

Shows all 6 `rdm_max` parameters (perceptual/memory noise, `a`, `t0`,
`w_0`, `w_s`) in three rows: population mean per group (natural units,
row 1), the dyscalc−control contrast itself with its own 94% HDI (row 2),
and between-subject SD τ per group (row 3, heterogeneity). Row 2 is
computed from `overview.json`'s `contrasts` field, not by subtracting
row 1's natural-unit intervals — that subtraction would be invalid: the
group means differ on a nonlinear (softplus) link, and the two groups'
posteriors are correlated, so only the model's own paired-draw contrast
(`dys − ctrl` per posterior draw, computed by `collect_overview.py`) gives
a valid CI. Rows 2–3 are therefore on the link/unconstrained scale, not
the same units as row 1 — labeled explicitly in the figure to avoid
misreading magnitudes across rows.

## `supp_rt_by_n1_group.{pdf,png}`

`plot_supp_rt_by_n1_group.py` (repo root, committed) ←
`data/magjudge_behavior_DNumRisk.csv` (gitignored, human-subjects). Model-free.

**Reconstruction (2026-11-12).** The original script (figure created
2026-08-17 12:52) was never committed and could not be found. The rebuilt
script reverse-engineers its recipe: 8 global quantile bins of log(n2/n1)
(`pd.qcut` over all trials), x = mean log(n2/n1) per bin, y = median RT
pooled over trials per (group, n1, bin). All 80 cell values match the
original figure to the plotted precision, including the two cells absent
for n1 = 5 and n1 = 7 (no trials in those bins). Only cosmetic differences
(house font sizes).

## `rdm_priors_revised.docx` (new, 2026-08-25)

`make_priors_docx.py` (kept HERE, next to its output, rather than at the repo
root: it is a one-off document generator, not part of the figure pipeline, and
the CLAUDE.md "lean repo" rule is about the latter. Keeping it at all is the
lesson from the missing-script incident below.) ← the two priors/estimation
paragraphs of `rdm_methods_revised.docx`, revised, in the same formatting
(Helvetica Neue 11 pt, justified, 1.5 spacing, matching page setup).

**Why it exists.** The methods draft stated a half-Cauchy(0.25) prior on all
group variance parameters, citing Gelman (2006), and quoted N(0.54, 0.3) /
N(−1.51, 0.5) / N(2.41, 0.5) population-mean priors for `a` / `t0` / `w_0`.
Both are wrong for the reported fits. Verified 2026-08-25 by rebuilding both
specs under bauer v0.4.0 and reading `estimation_model.free_RVs`: group SDs are
**HalfNormal**, scale `[0.25 control, 0.10 dyscalc]` for the two encoding-noise
parameters and `[0.25, 0.25]` for `a`/`t0`/`w_0`/`w_s`; all population means are
`Normal(0, 1)` per group. See `../MODELS.md` §4 for the table and for why the
cell-means design bypasses the package's intercept priors. The docx carries a
final italic "note to authors" paragraph flagging the asymmetric noise-SD prior
and its bearing on the memory-noise τ comparison; delete that paragraph before
submission.

## `rdm_methods_revised.docx`

Not a generated artifact — a manuscript methods-text draft, hand-edited
directly (via python-docx). The two edited paragraphs (DDM-vs-RDM model
comparison; priors and trial-exclusion accounting for the dynamic-model
parameters) now read correctly against the fitting code; the underlying
numbers (ΔELPD-LOO = 738, priors, 212/11,880 trial exclusion, `p_outlier`
contaminant) are documented permanently in `../MODELS.md` §4 rather than in
a session-scoped handover note.

## Process note

`CLAUDE.md` says "don't reintroduce one-off `make_*`/`plot_*` scripts;
they live in git history" — that assumes such scripts get committed at
least once before being pruned. The script behind `supp_ddm_vs_rdm` never
did, which is exactly how it got lost. Worth deciding explicitly: commit
`plot_supp_ddm_vs_rdm.py` (breaks the "lean repo" convention, but the
alternative already cost a rebuild) or intentionally discard it once the
figure is final, accepting that a future edit means rebuilding again. Not
changing `CLAUDE.md` unilaterally — flagging the tension for Gilles to
decide.

**2026-08-18:** `plot_supp_rdm_params.py` was added the same way — as a
plain, uncommitted file in the working tree (`git status` still shows both
`plot_supp_ddm_vs_rdm.py` and `plot_supp_rdm_params.py` as `??`). Same
tension, now for two files instead of one. Not committing either
unilaterally per the standing git policy (only commit when asked) — but
given the incident this file documents, leaving them uncommitted risks a
repeat.
