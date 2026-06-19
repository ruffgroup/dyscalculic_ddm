"""Posterior-predictive choice proportions for a fitted *_group model,
sliced by stake (n1) x order structure.

"Stake" = n1 (first / reference magnitude, held in memory); "order" is baked
into the model as memory noise (n1) vs perceptual noise (n2). Check: does
P(chose n2) vs log(n2/n1) shift with n1, and does the fitted model reproduce
that shift? Uses model.ppc (the supported regression path) -> simulated
choices, aggregated to proportions with a 94% band across posterior draws.

    ~/mambaforge/envs/bauer/bin/python ppc_choice.py \
        results/model-ddm_group_trace.nc --model ddm_group --n-draws 100 \
        --out notes/figures/ppc_stake_order_ddm.png
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fit_ddm import build_model

PAL = {"control": "#4C72B0", "dyscalc": "#C44E52"}


def load_df():
    df = pd.read_csv("data/magjudge_behavior_DNumRisk.csv")
    df = df[df["rt"] >= 0.20].copy()
    df["choice"] = df["choice"].astype(bool)
    df["chose_n2"] = df["chose_n2"].astype(bool)
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    # Unique index that still carries `subject` as a level (model.ppc joins on
    # the paradigm index, which must be unique).
    return df.set_index(["subject", "run", "trial_nr"])


def get_ppc(nc, model_label, n_draws, recompute=False):
    """Run (and cache) model.ppc, returning a long df with simulated_choice +
    trial covariates (one row per trial × ppc_sample)."""
    cache = f"results/ppc_{model_label}.parquet"
    if op.exists(cache) and not recompute:
        print(f"Using cached {cache}")
        return pd.read_parquet(cache)
    df = load_df()
    model = build_model(model_label, df)
    idata = az.from_netcdf(nc)
    ppc = model.ppc(df, idata, n_posterior_samples=n_draws, progressbar=True)
    ppc = ppc.reset_index().merge(
        df[["n1", "n2", "group", "chose_n2"]].reset_index(),
        on=["subject", "run", "trial_nr"], how="left")
    os.makedirs("results", exist_ok=True)
    ppc.to_parquet(cache)
    print(f"Cached {cache}")
    return ppc


def main(nc, model_label, n_draws, out, recompute):
    ppc = get_ppc(nc, model_label, n_draws, recompute)

    # Observed: per-subject P(chose n2) at each n1 -> CI across subjects.
    trial = ["subject", "run", "trial_nr"]
    obs_sub = (ppc.drop_duplicates(trial)
               .groupby(["subject", "group", "n1"], observed=True)["chose_n2"]
               .mean().reset_index())
    obs = obs_sub.groupby(["group", "n1"], observed=True)["chose_n2"].agg(
        m="mean", se=lambda x: x.std() / np.sqrt(len(x))).reset_index()

    # Model PPC: proportion per (group, n1, ppc_sample) -> mean + 94% HDI.
    per_sample = ppc.groupby(["group", "n1", "ppc_sample"], observed=True)[
        "simulated_choice"].mean().reset_index()
    pred = per_sample.groupby(["group", "n1"], observed=True)[
        "simulated_choice"].agg(
        mean="mean",
        lo=lambda x: az.hdi(x.values, hdi_prob=0.94)[0],
        hi=lambda x: az.hdi(x.values, hdi_prob=0.94)[1]).reset_index()

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for g in ["control", "dyscalc"]:
        o = obs[obs.group == g].sort_values("n1")
        pr = pred[pred.group == g].sort_values("n1")
        ax.errorbar(o.n1, o.m, yerr=o.se, fmt="o", color=PAL[g], capsize=3,
                    label=f"{g} (data)", zorder=3)
        ax.plot(pr.n1, pr["mean"], color=PAL[g], lw=2, zorder=2)
        ax.fill_between(pr.n1, pr.lo, pr.hi, color=PAL[g], alpha=.20,
                        label=f"{g} (model)")
    ax.axhline(0.5, ls=":", c="grey")
    ax.set_xlabel("n1 (first number / reference)")
    ax.set_ylabel("P(chose n2)")
    ax.set_title(f"PPC ({model_label}): choice proportion vs n1\n"
                 "points ± SE = data, line + band = model", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(op.dirname(op.abspath(out)), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("nc")
    p.add_argument("--model", default="ddm_group")
    p.add_argument("--n-draws", type=int, default=100)
    p.add_argument("--out", default="notes/figures/ppc_stake_order.png")
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    main(args.nc, args.model, args.n_draws, args.out, args.recompute)
