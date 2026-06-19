"""Fit a baseline Magnitude-Comparison DDM (bauer) to the DNumRisk data.

This is the "DDM0" model from numrisk's utils_02.build_model: a drift-diffusion
magnitude-comparison model with separate n1/n2 encoding noise, an independent
memory model, and a fixed (empirical) Bayesian-observer prior.

Why this fixes the "chains don't mix" problem (see
~/git/bauer/notes/fitting_ddm_models.md):
  * backend='numpyro' + chain_method='vectorized' instead of CPU pymc NUTS;
  * draws/tune passed by keyword (the old fit_bauer.py swapped them);
  * bauer's start-point finder (recommended_init='mapjitter') is ON by default
    for DDM models -- leave it on (do NOT pass find_init=False);
  * drop implausibly fast RTs (< 0.20 s) so the WFPT likelihood has gradient.

Smoke test (laptop, bauer conda env has jax/numpyro/hssm):
    ~/mambaforge/envs/bauer/bin/python fit_ddm.py \
        data/magjudge_behavior_subjectwise.csv \
        --n-subjects 2 --backend numpyro --tune 1000 --draws 1000

Production (cluster GPU, see run_fit.sh):
    python fit_ddm.py data/magjudge_behavior_subjectwise.csv \
        --n-subjects all --backend numpyro --tune 2000 --draws 1000 \
        --target-accept 0.99 --out trace.nc
"""
import argparse
import os.path as op

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from bauer.models import (DDMMagnitudeComparisonModel,
                           DDMMagnitudeComparisonRegressionModel,
                           MagnitudeComparisonRegressionModel)
from bauer.models.race import RaceDiffusionMagnitudeComparisonModel


class RaceMagnitudeComparisonRegressionModel(RaceDiffusionMagnitudeComparisonModel,
                                             MagnitudeComparisonRegressionModel):
    """Race-diffusion analog of DDMMagnitudeComparisonRegressionModel.

    bauer ships the DDM regression variant but not the race one. The concrete
    RaceDiffusionMagnitudeComparisonModel supplies the Wald-race likelihood and
    `_get_drifts` (resolved first in the MRO), but its __init__ doesn't accept
    `regressors`; so we route __init__ through MagnitudeComparisonRegressionModel
    (which sets up the patsy regression front-end) after stashing `advantage`.
    Threshold is the common absorbing bound `a`; drift uses the advantage
    decomposition (w_0, w_d, w_s).
    """

    def __init__(self, paradigm=None, regressors=None, advantage=True,
                 fit_w_s=True, fit_w_d=True, **kwargs):
        self.advantage = advantage
        self.fit_w_s = fit_w_s          # False ablates the OV/common-signal term
        self.fit_w_d = fit_w_d          # False FIXES discriminative gain w_d=1
        MagnitudeComparisonRegressionModel.__init__(
            self, paradigm, regressors=regressors, **kwargs)


def build_model(model_label, df, fit_prior=False):
    # Generic spec: "<arch>:<comma-separated params to regress on group>", e.g.
    # "rdm:perceptual_noise_sd,memory_noise_sd,a,t0" or "ddm:t0". fit_prior is
    # turned on automatically if any prior_* parameter is regressed. v_scale is
    # never fit. This is what the model-grid sweep uses.
    if ":" in model_label:
        arch, _, reg_str = model_label.partition(":")
        regs = [r for r in reg_str.split(",") if r]
        fp = fit_prior or any(r.startswith("prior") for r in regs)
        # If n1/n2_evidence_sd are named, use the INDEPENDENT noise model
        # (separate per-stimulus encoding noise); else shared_perceptual_noise.
        indep = any(r in ("n1_evidence_sd", "n2_evidence_sd") for r in regs)
        # DEFAULT = cell-means: fixed_regressors AND random_regressors both
        # '0 + C(group)', i.e. a separate distribution per group (own population
        # mean AND own between-subject SD). This is the correct parameterization
        # for a between-subjects factor -- no random-slope-on-contrast confound,
        # no forced homoscedasticity. The group effect is the derived contrast
        # dyscalc - control. GROUP_PARAM=legacy restores the old
        # regressors={p:'group'} (Intercept + group random slope).
        import os as _os
        if _os.environ.get("GROUP_PARAM") == "legacy":
            reg_kw = dict(regressors={r: "group" for r in regs})
        else:
            # random_regressors='0 + C(group)' alone -> bauer auto-promotes it to
            # the fixed design (per-group means) + per-group SD: separate
            # distribution per group, the clean between-subjects parameterization.
            reg_kw = dict(random_regressors={r: "0 + C(group)" for r in regs})
        kw = dict(**reg_kw, fit_prior=fp,
                  fit_separate_evidence_sd=True,
                  memory_model="independent" if indep else "shared_perceptual_noise")
        # arch 'rdm_nows' = race model with the w_s (sum / overall-magnitude)
        # drift term ablated, so the n1 effect comes only from shrinkage.
        if arch == "rdm_nows":
            return RaceMagnitudeComparisonRegressionModel(df, fit_w_s=False, **kw)
        # arch 'rdm_nowd' = race with the discriminative gain w_d FIXED to 1
        # (the v_scale analog), so the encoding noise/boundary set the scale and
        # σ becomes comparable to the DDM. See notes/why_noise_attribution_flips.md.
        if arch == "rdm_nowd":
            return RaceMagnitudeComparisonRegressionModel(df, fit_w_d=False, **kw)
        # arch 'rdm_nows_nowd' = race WITHOUT the overall-magnitude term (w_s≡0)
        # AND with the discriminative gain fixed (w_d≡1). The −w_s control on
        # the canonical w_d=1 footing.
        if arch == "rdm_nows_nowd":
            return RaceMagnitudeComparisonRegressionModel(
                df, fit_w_s=False, fit_w_d=False, **kw)
        # arch 'ddm_sv' = DDM with memory noise routed to across-trial drift
        # variability (Ratcliff sv) instead of the drift SNR. Needs the shared
        # (perceptual/memory) noise split; see notes/sv_model_explained.md.
        if arch == "ddm_sv":
            kw["memory_model"] = "shared_perceptual_noise"
            return DDMMagnitudeComparisonRegressionModel(df, memory_as_sv=True, **kw)
        cls = {"choice": MagnitudeComparisonRegressionModel,
               "ddm": DDMMagnitudeComparisonRegressionModel,
               "rdm": RaceMagnitudeComparisonRegressionModel}[arch]
        return cls(df, **kw)

    """Map a short label to a configured bauer model.

    The three *_group models share the same cognitive front-end (split
    perceptual vs memory noise, both regressed on group) and differ ONLY in
    the response/likelihood layer — so they isolate what the choice/RT
    mechanism implies about where the dyscalculia deficit sits:

    choice_group : static cumulative-normal choice (NO accumulation). The
                   "simple choice model"; only choices, no RT, no threshold.
    ddm_group    : drift-diffusion (WFPT). Adds threshold `a ~ group`.
    rdm_group    : race-diffusion (Wald). Threshold `a ~ group`, advantage
                   drift (w_0, w_d, w_s).

    ddm0 is the independent-noise baseline (no group contrast), kept for
    reference. `fit_prior=True` estimates the Bayesian-observer prior and
    lets it (prior_mu, prior_sd) also vary by group.
    """
    # `*_full` = the maximally flexible model: prior estimated and EVERYTHING
    # (noise, threshold, non-decision time, prior) regressed on group. Fit it,
    # see which contrasts are credible, then constrain back to `*_group`.
    full = model_label.endswith("_full")
    if full:
        fit_prior = True

    noise_reg = {"perceptual_noise_sd": "group", "memory_noise_sd": "group"}
    prior_reg = {"prior_mu": "group", "prior_sd": "group"} if fit_prior else {}
    acc_reg = {"a": "group"}                 # decision threshold (DDM/RDM only)
    if full:
        acc_reg["t0"] = "group"              # non-decision time too
    common = dict(fit_prior=fit_prior, fit_separate_evidence_sd=True,
                  memory_model="shared_perceptual_noise")

    if model_label == "ddm0":
        return DDMMagnitudeComparisonModel(
            df, fit_prior=False, fit_separate_evidence_sd=True,
            memory_model="independent")
    if model_label in ("choice_group", "choice_full"):   # static: no a / t0
        return MagnitudeComparisonRegressionModel(
            df, regressors={**noise_reg, **prior_reg}, **common)
    if model_label in ("ddm_group", "ddm_full"):
        return DDMMagnitudeComparisonRegressionModel(
            df, regressors={**noise_reg, **acc_reg, **prior_reg}, **common)
    if model_label == "ddm_sv_group":
        # Like ddm_group, but memory noise -> across-trial drift variability
        # (Ratcliff sv) so perceptual/memory are separately identifiable.
        # See notes/sv_model_explained.md.
        return DDMMagnitudeComparisonRegressionModel(
            df, regressors={**noise_reg, **acc_reg}, memory_as_sv=True, **common)
    if model_label in ("rdm_group", "rdm_full"):
        return RaceMagnitudeComparisonRegressionModel(
            df, regressors={**noise_reg, **acc_reg, **prior_reg}, **common)
    # --- single-DoF arbiters (each = ddm_group + ONE extra group term) ----
    if model_label == "ddm_t0group":
        # Does the dyscalc RT slowing load onto non-decision time instead of
        # perceptual noise? ddm_group + t0 ~ group.
        return DDMMagnitudeComparisonRegressionModel(
            df, regressors={**noise_reg, "a": "group", "t0": "group"},
            fit_prior=False, fit_separate_evidence_sd=True,
            memory_model="shared_perceptual_noise")
    # NB: we deliberately do NOT fit v_scale (the drift slope). Freeing it
    # reintroduces a perfect v_scale<->evidence_sd degeneracy (ddm.py:116-124)
    # and destabilizes the fit; fit_v_scale=False is kept everywhere.
    raise ValueError(f"unknown model_label: {model_label!r}")


def main(csv, model_label, n_subjects, backend, draws, tune, target_accept,
         seed, out, chains=4, consistent_choice_noise=False, init=None,
         halfnormal=False, p_outlier=None, fit_prior=False):
    df = pd.read_csv(csv)

    # bauer needs: choice (bool, True == chose n2), rt (s, > 0), n1/n2,
    # `subject` in the index/column, and (for ddm_group) a `group` column.
    df["choice"] = df["choice"].astype(bool)

    n_before = len(df)
    df = df[df["rt"] >= 0.20].copy()
    print(f"Dropped {n_before - len(df)} trials with rt < 0.20 s "
          f"({len(df)} remaining)")

    df = df.set_index("subject")

    if n_subjects != "all":
        keep = sorted(df.index.unique())[: int(n_subjects)]
        df = df.loc[keep]
    print(f"Fitting model {model_label!r}: {df.index.nunique()} subjects, "
          f"{len(df)} trials")

    model = build_model(model_label, df, fit_prior=fit_prior)
    # KLW/DDM-consistent choice noise (posterior-mean SD) instead of the raw
    # evidence_sd default. Only affects static choice models; accumulators
    # already normalize by posterior_mean_sd.
    if consistent_choice_noise:
        model.consistent_choice_noise = True
        print("consistent_choice_noise=True (KLW-consistent choice normalization)")
    if halfnormal:
        model.group_sd_dist = "halfnormal"
        print("group_sd_dist=halfnormal (tames the HalfCauchy runaway tail)")
    # RT outlier contaminant (DDM/RDM only; base classes carry p_outlier=0.05).
    # Set 0.0 for pure WFPT/Wald (no contaminant) -- used to isolate whether the
    # contaminant helps or hurts a given accumulator.
    if p_outlier is not None and hasattr(model, "p_outlier"):
        model.p_outlier = p_outlier
        model.free_parameters = model.get_free_parameters()
        print(f"p_outlier={p_outlier} (contaminant {'OFF' if p_outlier == 0 else 'fixed'})")
    model.build_estimation_model(data=df, hierarchical=True)

    # recommended_init='mapjitter' is the default for DDM models -> dispersed,
    # data-informed starting points. This is the lever that makes it mix.
    sample_kw = {}
    if init is not None:
        sample_kw["find_init"] = init      # e.g. 'pathfinder' or 'mapjitter'
        print(f"init strategy: {init}")
    idata = model.sample(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        backend=backend,
        random_seed=seed,
        **sample_kw,
    )

    # Log-likelihood is no longer computed automatically (needed for LOO/WAIC).
    with model.estimation_model:
        pm.compute_log_likelihood(idata)

    az.to_netcdf(idata, out)
    print(f"Saved trace -> {out}")

    # --- Convergence diagnostics: the whole point of this exercise ----------
    summ = az.summary(idata)
    n_div = int(idata.sample_stats["diverging"].sum()) \
        if "diverging" in idata.sample_stats else -1
    print("\n=== Diagnostics ===")
    print(f"max r_hat : {summ['r_hat'].max():.4f}")
    print(f"min ess_bulk: {summ['ess_bulk'].min():.0f}")
    print(f"min ess_tail: {summ['ess_tail'].min():.0f}")
    print(f"divergences : {n_div}")
    worst = summ.sort_values("r_hat", ascending=False).head(8)
    print("\nWorst-mixing parameters:")
    print(worst[["mean", "sd", "r_hat", "ess_bulk"]])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--model", default="ddm0",
                   help="named (ddm0, choice_group, ddm_group, rdm_group, "
                        "ddm_t0group, *_full) OR a generic spec "
                        "'<arch>:<param,param,...>' e.g. "
                        "'rdm:perceptual_noise_sd,memory_noise_sd,a,t0'")
    p.add_argument("--n-subjects", default="all",
                   help="'all' or an integer (for smoke tests)")
    p.add_argument("--backend", default="numpyro",
                   choices=["pymc", "numpyro", "blackjax"])
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--target-accept", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--out", default="trace.nc")
    p.add_argument("--consistent-choice-noise", action="store_true",
                   help="KLW-consistent posterior-mean-SD choice noise "
                        "(static choice models only)")
    p.add_argument("--init", default=None,
                   help="start-point finder: 'pathfinder' (multipath, for "
                        "multimodal DDM posteriors) or 'mapjitter' (default).")
    p.add_argument("--halfnormal", action="store_true",
                   help="HalfNormal group-SD prior instead of HalfCauchy "
                        "(tames the runaway-variance funnel).")
    p.add_argument("--p-outlier", type=float, default=None,
                   help="DDM/RDM RT outlier contaminant rate. 0.0 = pure "
                        "WFPT/Wald (no contaminant); default leaves the model's "
                        "own default (0.05 on the lapse-enabled base classes).")
    p.add_argument("--fit-prior", action="store_true",
                   help="Estimate the observer prior (prior_mu, prior_sd) as free "
                        "parameters, SHARED across groups (not regressed on group). "
                        "Default: prior fixed to the empirical stimulus distribution.")
    args = p.parse_args()

    main(args.csv, args.model, args.n_subjects, args.backend, args.draws,
         args.tune, args.target_accept, args.seed, args.out, args.chains,
         args.consistent_choice_noise, args.init, args.halfnormal,
         args.p_outlier, args.fit_prior)
