"""Posterior-predictive compression check for the key models.

Compression = regression-to-the-prior: as n1 grows, P(chose n2) falls, and a
noisier observer's curve is flatter (pulled toward the prior mean). We show
observed vs model-predicted P(chose n2) as a function of n1, split by group,
one panel per key model.

  ~/mambaforge/envs/bauer/bin/python make_key_ppc.py
"""
import os.path as op
import warnings

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle
import fit_ddm

warnings.filterwarnings("ignore")
figstyle.apply()
HERE = op.dirname(op.abspath(__file__))
GROUP = figstyle.GROUP                       # control blue, dyscalc red
N_PPC = 60

# (panel title, build tag, KLW?, converged trace)
KEY = [
    ("Choice (KLW)", "choice_group", True,  "results/model-choice_group_klw_trace.nc"),
    ("DDM",          "ddm_group",    False, "results/model-ddm_group_long_trace.nc"),
    ("RDM",          "rdm_group",    False, "results/model-rdm_group_long_trace.nc"),
]


def load_df():
    df = pd.read_csv(op.join(HERE, "data", "magjudge_behavior_DNumRisk.csv"))
    df = df[df["rt"] >= 0.20].copy()
    df["choice"] = df["chose_n2"].astype(bool)
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    # binned log-ratio for the psychometric view (decision variable)
    df["lr_raw"] = np.log(df["n2"] / df["n1"])
    edges = np.quantile(df["lr_raw"], np.linspace(0, 1, 10))
    edges = np.unique(edges)
    df["lr"] = pd.cut(df["lr_raw"], edges, include_lowest=True).apply(
        lambda iv: round(0.5 * (iv.left + iv.right), 3))
    return df.set_index(["subject", "run", "trial_nr"])


def ppc_choices(df, tag, klw, trace):
    """Posterior-predictive simulated_choice joined to n1/group/ratio bin."""
    m = fit_ddm.build_model(tag, df)
    if klw:
        m.consistent_choice_noise = True
    m.build_estimation_model(df, hierarchical=True)
    idata = az.from_netcdf(op.join(HERE, trace))
    sim = m.ppc(df, idata, n_posterior_samples=N_PPC, progressbar=False)
    return sim.join(df[["n1", "group", "lr"]])


def band(sim, xvar):
    """Median + 94% HDI of P(chose n2) across ppc samples, per (group, xvar)."""
    per = (sim.groupby(["ppc_sample", "group", xvar])["simulated_choice"]
              .mean().reset_index())
    rows = []
    for (g, x), s in per.groupby(["group", xvar]):
        v = s["simulated_choice"].values
        lo, hi = az.hdi(v, hdi_prob=0.94)
        rows.append(dict(group=g, x=x, med=np.median(v), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def make_panel(ax, obs, pred, xcol, title):
    for g, col in GROUP.items():
        pr = pred[pred.group == g].sort_values("x")
        ob = obs[obs.group == g].sort_values(xcol)
        ax.fill_between(pr.x, pr.lo, pr.hi, color=col, alpha=0.22, lw=0)
        ax.plot(pr.x, pr.med, color=col, lw=1.3, zorder=3)
        ax.scatter(ob[xcol], ob.p, s=24, facecolor="white", edgecolor=col,
                   lw=1.3, zorder=4)
    ax.axhline(0.5, ls=":", c="0.6", lw=0.8)
    ax.set_title(title, fontsize=9.5)


def main():
    df = load_df()
    # compute PPC choices once per model (the expensive step)
    sims = {title: ppc_choices(df, tag, klw, trace)
            for title, tag, klw, trace in KEY}

    # ---- Figure 1: compression, P(chose n2) vs n1 ----
    obs_n1 = (df.reset_index().groupby(["group", "n1"])["choice"].mean()
                .rename("p").reset_index())
    fig, axes = plt.subplots(1, len(KEY), figsize=(7.6, 2.7), sharey=True,
                             constrained_layout=True)
    for ax, (title, *_), in zip(axes, KEY):
        make_panel(ax, obs_n1, band(sims[title], "n1"), "n1", title)
        ax.set_xticks([5, 10, 15, 20]); ax.set_xlabel("n1 (first number)")
    axes[0].set_ylabel("P(chose n2)"); axes[0].set_ylim(0.25, 0.65)
    axes[0].annotate("Control", (0.05, 0.16), xycoords="axes fraction",
                     color=GROUP["control"], fontsize=8, fontweight="bold")
    axes[0].annotate("Dyscalc", (0.05, 0.05), xycoords="axes fraction",
                     color=GROUP["dyscalc"], fontsize=8, fontweight="bold")
    fig.suptitle("Posterior-predictive compression:  P(chose n2) vs n1   "
                 "(points = data, line/band = model 94% PPC)", fontsize=9.5)
    figstyle.finish(fig, op.join(HERE, "notes", "figures", "key_models_ppc_compression.pdf"))

    # ---- Figure 2: psychometric, P(chose n2) vs log(n2/n1) (clean Weber view) ----
    obs_lr = (df.reset_index().groupby(["group", "lr"])["choice"].mean()
                .rename("p").reset_index())
    fig2, axes2 = plt.subplots(1, len(KEY), figsize=(7.6, 2.7), sharey=True,
                               constrained_layout=True)
    for ax, (title, *_), in zip(axes2, KEY):
        make_panel(ax, obs_lr, band(sims[title], "lr"), "lr", title)
        ax.axvline(0, ls=":", c="0.6", lw=0.8)
        ax.set_xlabel("log(n2 / n1)")
    axes2[0].set_ylabel("P(chose n2)"); axes2[0].set_ylim(0, 1)
    axes2[0].annotate("Control", (0.05, 0.88), xycoords="axes fraction",
                      color=GROUP["control"], fontsize=8, fontweight="bold")
    axes2[0].annotate("Dyscalc", (0.05, 0.78), xycoords="axes fraction",
                      color=GROUP["dyscalc"], fontsize=8, fontweight="bold")
    fig2.suptitle("Psychometric function:  P(chose n2) vs log(n2/n1)   "
                  "(slope = acuity; flatter = more compression)", fontsize=9.5)
    figstyle.finish(fig2, op.join(HERE, "notes", "figures", "key_models_ppc_psychometric.pdf"))


if __name__ == "__main__":
    main()
