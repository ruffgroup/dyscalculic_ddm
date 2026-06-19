# Proposal: separating fixed and random effects in bauer regression models

**Status:** ✅ PROTOTYPED & TESTED on bauer branch `random-effects-api`
(worktree `/private/tmp/bauer-re-experiment`, commit d597485 — `core.py` +124,
`magnitude.py` +9; NOT merged). Smoke tests pass:
- legacy `regressors={p:'C(group)'}` → `DeprecationWarning`, offset still over
  `[Intercept, group]` (old graph bit-for-bit);
- new `fixed_regressors={p:'C(group)'}` (no random) → offset over a `_re` coord
  = `['Intercept']` only, mean over `[Intercept, group]` (random-intercept +
  fixed contrast); samples end-to-end (numpyro), contrast recovered as fixed;
- `random_regressors={p:'C(group)'}` → between-subjects `UserWarning` fires.

(Still independent of the convergence fix — that's the HalfNormal prior, tested
separately.)

## The bug, precisely
`RegressionModel.build_hierarchical_nodes` (bauer `core.py` ~969–1015):
```python
group_mu       = Normal(f"{name}_mu",     dims=(f'{name}_regressors',))
group_sd       = _group_sd_node(name, ..., dims=(f'{name}_regressors',))
subject_offset = Normal(f'{name}_offset',  dims=('subject', f'{name}_regressors'))
param_subj     = group_mu + group_sd * subject_offset            # then @ design matrix
```
`subject_offset` and `group_sd` span the **full** regressor dim, so *every* design
column gets a per-subject random SD — including a **between-subjects** contrast
like `group[T.dyscalc]`. Consequences:
- the ~33 control subjects have that design column = 0 → their offset on it is
  multiplied by 0 → **non-identified nuisance dimensions** (sample the prior);
- dyscalc subjects gain an extra random-effect column → modelled as
  **heteroscedastically more variable** than controls. Unintended.

(Empirically the dyscalc-effect SDs are tiny ~0.06 while the *intercept* SD runs
away — so this is a correctness/cleanliness issue, not the main driver of the
ESS≈7 collapse, which is the HalfCauchy tail. Fix the prior for convergence;
fix this for correctness.)

## IMPORTANT correction: `regressors` already carries random effects
`build_hierarchical_nodes` builds, per design column,
`param_subj[reg] = group_mu[reg] + group_sd[reg]·offset[subj,reg]` — i.e. **every**
`regressors` term is a *random coefficient* (fixed population mean **plus** a
per-subject random deviation). So `regressors` is NOT "the fixed design"; for
`group[T.dyscalc]` it currently fits a fixed contrast **and** a random slope.
Therefore **defaulting random effects to intercept-only would NOT be
backwards-compatible** — it would delete random effects that exist today.

## Recommended API — a sibling `random_regressors` keyword
```python
Model(df,
      regressors={'perceptual_noise_sd': 'C(group)'},        # population-mean design (unchanged)
      random_regressors={'perceptual_noise_sd': '1'})        # which terms ALSO vary by subject
```
- `regressors` = which terms get a **population mean** (its current role; it just
  no longer *implies* a random effect on every term).
- `random_regressors` = which of those terms additionally carry a **per-subject
  random effect**. **Default = mirror `regressors`** (≡ today's behavior, exact
  back-compat). Set `{p: '1'}` for the standard random-intercept + fixed
  between-subjects-contrast model.
- Implementation: build a *second* patsy design matrix from the random formula;
  `group_sd`/`subject_offset` get dims over the **random** columns only;
  `param_subj = mu @ dm_fixed + (group_sd * offset) @ dm_random`.

### Why this over the alternatives
- **vs nested `{param: {'fixed':…, 'random':…}}`** (lme4-style): more familiar but
  a bigger, breakier change to the `regressors` contract and everything that
  iterates `self.regressors` / `self.design_matrices`.
- **vs a coarse `random_effects='intercept'|'all'` switch**: can't express
  "random slope on a *within-subject* covariate but fixed *between-subjects*
  contrast" within one parameter.

### Within- vs between-subject (the genuinely tricky part)
Random slopes are legitimate for **within-subject** covariates (ISI, session,
trial-type) and are the *bug* only for **between-subject** ones (group). Make the
user declare random structure explicitly via `random_regressors` (explicit beats
magic), **plus** a validation guard: if a requested random column is constant
within every subject (⇒ between-subjects), warn/raise. Catches the mistake
without forcing detection-only behaviour.

### Heteroscedastic-by-group (optional, principled)
A separate group-SD formula, e.g. `random_sd_regressors={param: 'C(group)'}` →
`group_sd` indexed by group. This is the *clean* way to ask "are dyscalcs more
heterogeneous?" — the principled version of what the code does by accident now.

### Backward compatibility (the crux)
Current behaviour = a random effect on **every** `regressors` term. So
`random_regressors` MUST default to "match `regressors`" to reproduce existing
graphs exactly; intercept-only is **opt-in**. Defaulting to intercept-only would
be a **breaking change** (silently drops random slopes, changes every posterior)
— acceptable only behind a major-version bump + migration note. Recommendation:
ship with the mirror default, document `random_regressors={p:'1'}` as the
preferred structure for between-subjects designs, and reconsider the default
later.

## Prototype / smoke-test plan (when built)
Worktree off the current branch; verify the model graph: `subject_offset` dims
only over the random columns (intercept), `group` enters as a fixed `Normal`
coefficient, old API reproduces the old graph; then a tiny `sample()` (6
subjects) to confirm end-to-end.

---
*Design analysis by a delegated agent (reading-only; prototype blocked on
sandboxed git/Bash). Implementation is straightforward for someone with repo
write access.*
