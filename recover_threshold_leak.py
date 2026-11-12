"""Does the RDM's *threshold* difference end up inside the choice-only PCM's
*noise* parameters? An interventional (do-operator) simulation.

`recover_cross_pcm.py` showed the choice-only PCM reproduces the paradoxical
dyscalc<control perceptual-noise contrast when data are simulated from the
real rdm_max posterior (82% of reps flip sign) -- but that is an *observational*
result: every simulated world had BOTH a higher dyscalc noise AND a higher
dyscalc threshold, so it cannot say which one the PCM is mis-reading. (Regressing
the recovered contrast on the per-draw true contrasts across those 997 reps is
underpowered by design: the posterior's own spread in the threshold contrast is
tiny (SD 0.10) next to its offset (mean +0.37) and never crosses zero, so there
is no counterfactual in the sample.)

This script *intervenes*. For one posterior draw it rescales individual group
contrasts on the link scale before simulating -- leaving everything else in that
draw untouched -- then refits the choice-only PCM to the simulated choices:

  full     everything as fitted                       (reference; = recover_cross_pcm)
  a_x0     threshold contrast set to 0                (compensation removed)
  a_x0.5   threshold contrast halved                  } dose-response
  a_x1.5   threshold contrast x 1.5                   }
  w0_x0    urgency (w_0) contrast set to 0            (specificity: is it *a*?)
  noise_x0 BOTH noise contrasts set to 0, decision side kept  (pure leak)
  all_x0   every group contrast set to 0              (pipeline null control)

Mechanics: in the cell-means parameterization the per-subject coefficient array
`{p}` has dims (chain, draw, subject, {p}_regressors) and lives on the *link*
(pre-softplus) scale -- bauer's regression path applies the design matrix first
and the transform after (`RegressionMixin.get_trialwise_variable`). A subject's
design row is one-hot for their group, so adding (scale-1)*Delta to the
`C(group)[dyscalc]` column rescales exactly the group contrast, preserving every
subject's deviation from their own group mean, and is a no-op for controls.
`RaceMixin.ppc` simulates straight from that array, so the intervention is exact.

One task = one (rep, condition):
  python recover_threshold_leak.py --task 0 \
      --trace /shares/zne.uzh/gdehol/dyscalculic_ddm/model-rdm_max_cont_hn_trace.nc
Cluster: `sbatch --array=0-279%120 run_recover_threshold_leak.sh <trace>`
(rep = task // 7, condition = CONDITIONS[task % 7]; rep k uses the SAME
posterior draw as rep k of recover_cross_pcm.py -- identical seed/shuffle.)
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

import fit_ddm
from recover_cross_pcm import (TRUTH_SPEC, REFIT_SPEC, CONTRASTS,
                               load_df, pick_chain_draw, group_contrast)

HERE = op.dirname(op.abspath(__file__))

# Which group contrasts survive, and at what multiple of their fitted value.
# Keys not listed are left exactly as sampled.
CONDITIONS = {
    "full":     {},
    "a_x0":     {"a": 0.0},
    "a_x0.5":   {"a": 0.5},
    "a_x1.5":   {"a": 1.5},
    "w0_x0":    {"w_0": 0.0},
    "noise_x0": {"perceptual_noise_sd": 0.0, "memory_noise_sd": 0.0},
    # NB: not named "null" -- pandas.read_csv would read that string back as NaN.
    "all_x0":   {p: 0.0 for p in CONTRASTS},
}
COND_NAMES = list(CONDITIONS)


def _levels(da):
    """(regressor dim, control level, dyscalc level) of a cell-means parameter."""
    rdim = [d for d in da.dims if d.endswith("_regressors")][0]
    lev = [str(c) for c in da.coords[rdim].values]
    return rdim, next(c for c in lev if "control" in c), next(c for c in lev if "dyscalc" in c)


def rescale_contrast(post, param, scale):
    """Multiply `param`'s dyscalc-control contrast by `scale`, on the link scale.

    Adds (scale-1)*Delta to the dyscalc column of BOTH the group mean `{p}_mu`
    and the per-subject coefficients `{p}` (only the latter is read by ppc; the
    former is kept in sync for bookkeeping). Returns (Delta_fitted, shift)."""
    mu = post[f"{param}_mu"]
    rdim, ctrl, dys = _levels(mu)
    delta = float((mu.sel({rdim: dys}) - mu.sel({rdim: ctrl})).values.ravel()[0])
    shift = (scale - 1.0) * delta
    onehot = xr.DataArray(
        (np.array([str(c) for c in mu.coords[rdim].values]) == dys).astype(float),
        dims=(rdim,), coords={rdim: mu.coords[rdim]})
    post[f"{param}_mu"] = mu + shift * onehot
    post[param] = post[param] + shift * onehot
    return delta, shift


def realized(post, param, is_dys):
    """What the simulated world actually has, from the per-subject coefficients
    each subject's own design row selects: (link-scale group contrast, natural
    control mean, natural dyscalc mean). This is the finite-sample truth the
    refit sees -- not the group-mu contrast."""
    da = post[param]
    rdim, ctrl, dys = _levels(da)
    ctrl_col = da.sel({rdim: ctrl}).values.ravel()
    dys_col = da.sel({rdim: dys}).values.ravel()
    link = np.where(is_dys, dys_col, ctrl_col)
    softplus = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    nat = softplus(link) if param != "w_s" else link      # w_s has an identity link
    return (float(link[is_dys].mean() - link[~is_dys].mean()),
            float(nat[~is_dys].mean()), float(nat[is_dys].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="real rdm_max posterior .nc")
    ap.add_argument("--task", type=int, required=True,
                    help="rep*len(CONDITIONS) + condition index")
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--backend", default="numpyro")
    ap.add_argument("--target-accept", type=float, default=0.95)
    args = ap.parse_args()

    rep, cond = divmod(args.task, len(COND_NAMES))
    cond = COND_NAMES[cond]
    scales = CONDITIONS[cond]
    df = load_df()

    truth_model = fit_ddm.build_model(TRUTH_SPEC, df)
    truth_model.build_estimation_model(df, hierarchical=True)   # ppc needs it built
    idata = az.from_netcdf(args.trace)
    c, d = pick_chain_draw(idata.posterior.sizes["chain"],
                           idata.posterior.sizes["draw"], rep)

    # isel returns fresh Datasets, and every intervention below assigns a NEW
    # array (`post[p] = post[p] + shift`), so the on-disk trace is untouched.
    idata_k = idata.isel(chain=[c], draw=[d])
    post = idata_k.posterior
    subj_group = df.reset_index().groupby("subject")["group"].first()
    is_dys = np.array([str(subj_group.loc[s]) == "dyscalc"
                       for s in post.coords["subject"].values])

    row = {"task": args.task, "rep": rep, "cond": cond, "chain": c, "draw": d,
           "n_dys": int(is_dys.sum()), "n_ctrl": int((~is_dys).sum())}
    # ---- intervene, and record what the simulated world now contains ----
    for p in CONTRASTS:
        if p in scales:
            delta, shift = rescale_contrast(post, p, scales[p])
            row[f"scale_{p}"] = scales[p]
        else:
            delta = float(group_contrast(post[f"{p}_mu"]).values.ravel()[0])
            row[f"scale_{p}"] = 1.0
        row[f"fitted_{p}"] = delta                      # the draw's own contrast
        link, nat_c, nat_d = realized(post, p, is_dys)
        row[f"true_{p}"] = link                         # after intervention
        row[f"nat_{p}_control"], row[f"nat_{p}_dyscalc"] = nat_c, nat_d

    # ---- simulate choices+RTs from the (possibly intervened) draw ----
    # Same seed across conditions within a rep -> paired comparison.
    sim = truth_model.ppc(df, idata_k, n_posterior_samples=1, progressbar=False,
                          random_seed=12345 + rep)
    sim = sim.reset_index().merge(
        df[["n1", "n2", "group"]].reset_index(),
        on=["subject", "run", "trial_nr"], how="left")
    sim["choice"] = sim["simulated_choice"].astype(bool)
    sim["rt"] = sim["simulated_rt"].astype(float)
    sim = sim[sim["rt"] >= 0.20].copy()          # same trial-exclusion real data got
    sim["correct"] = sim["choice"] == (sim["n2"] > sim["n1"])
    row["n_trials"] = int(len(sim))
    row["sim_acc"] = float(sim["correct"].mean())
    for g in ("control", "dyscalc"):
        sg = sim[sim["group"] == g]
        row[f"sim_acc_{g}"] = float(sg["correct"].mean())
        row[f"sim_rt_{g}"] = float(sg["rt"].median())

    # ---- refit the choice-only PCM (no threshold) to the simulated CHOICES ----
    fit_df = sim[["subject", "run", "trial_nr", "n1", "n2", "choice", "rt", "group"]]
    fit_df = fit_df.set_index(["subject", "run", "trial_nr"])
    refit_model = fit_ddm.build_model(REFIT_SPEC, fit_df)
    refit_model.build_estimation_model(fit_df, hierarchical=True)
    ridata = refit_model.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                                target_accept=args.target_accept, backend=args.backend)
    rpost = ridata.posterior
    row["max_rhat"] = float(az.summary(ridata)["r_hat"].max())
    for p in ("perceptual_noise_sd", "memory_noise_sd"):
        vals = group_contrast(rpost[f"{p}_mu"]).values.ravel()
        row[f"rec_{p}"] = float(vals.mean())
        row[f"recP_{p}"] = float((vals > 0).mean())
        lo, hi = az.hdi(vals, hdi_prob=0.94)
        row[f"recLo_{p}"], row[f"recHi_{p}"] = float(lo), float(hi)

    out_dir = op.join(HERE, "results", "recover_threshold_leak")
    os.makedirs(out_dir, exist_ok=True)
    out = op.join(out_dir, f"rep{rep:03d}_{cond}.csv")
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"[{cond}] rep {rep} (chain {c} draw {d}): "
          f"TRUE a {row['true_a']:+.3f} (nat {row['nat_a_control']:.2f}->{row['nat_a_dyscalc']:.2f})  "
          f"perc {row['true_perceptual_noise_sd']:+.3f}  "
          f"acc {row['sim_acc_control']:.3f}/{row['sim_acc_dyscalc']:.3f}  "
          f"-> REFIT-PCM perc {row['rec_perceptual_noise_sd']:+.3f} "
          f"(P={row['recP_perceptual_noise_sd']:.2f}) mem {row['rec_memory_noise_sd']:+.3f}  "
          f"rhat={row['max_rhat']:.2f} -> {out}")


if __name__ == "__main__":
    main()
