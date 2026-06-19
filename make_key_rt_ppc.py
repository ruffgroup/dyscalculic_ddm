"""Posterior-predictive RT check for the RT models.

Two views, observed (points) vs model PPC (line + 94% band), split by group:
  (1) RT quantiles (.1/.3/.5/.7/.9) — does the model capture the whole RT
      distribution per group, including the group RT difference?
  (2) RT quantiles split by correct vs error — the slow-error signature
      (memory / across-trial variability) lives here.

Choice model has no RT, so only DDM / RDM appear.

  ~/mambaforge/envs/bauer/bin/python make_key_rt_ppc.py
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
GROUP = figstyle.GROUP
N_PPC = 60
QS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

KEY = [
    ("DDM", "ddm_group", "results/model-ddm_group_long_trace.nc"),
    ("RDM", "rdm_group", "results/model-rdm_group_long_trace.nc"),
]


def load_df():
    df = pd.read_csv(op.join(HERE, "data", "magjudge_behavior_DNumRisk.csv"))
    df = df[df["rt"] >= 0.20].copy()
    df["choice"] = df["chose_n2"].astype(bool)
    df["correct"] = (df["chose_n2"].astype(bool) == (df["n2"] > df["n1"]))
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    return df.set_index(["subject", "run", "trial_nr"])


def ppc_rt(df, tag, trace):
    m = fit_ddm.build_model(tag, df)
    m.build_estimation_model(df, hierarchical=True)
    idata = az.from_netcdf(op.join(HERE, trace))
    sim = m.ppc(df, idata, n_posterior_samples=N_PPC, progressbar=False)
    sim = sim.join(df[["group", "n1", "n2"]])
    sim["correct"] = (sim["simulated_choice"].astype(bool) == (sim["n2"] > sim["n1"]))
    sim = sim[sim["simulated_rt"] >= 0.20]
    return sim


def band(sim, mask_col=None, mask_val=None):
    """Per-group RT quantile band across ppc samples; optional correct/error mask."""
    s = sim if mask_col is None else sim[sim[mask_col] == mask_val]
    out = {}
    for g in GROUP:
        sg = s[s.group == g]
        # quantiles within each ppc sample, then HDI across samples
        qd = sg.groupby("ppc_sample")["simulated_rt"].quantile(QS).unstack()
        med = qd.median(axis=0).values
        lo = np.array([az.hdi(qd[c].values, hdi_prob=0.94)[0] for c in qd.columns])
        hi = np.array([az.hdi(qd[c].values, hdi_prob=0.94)[1] for c in qd.columns])
        out[g] = (med, lo, hi)
    return out


def obs_q(df, mask_col=None, mask_val=None):
    d = df if mask_col is None else df[df[mask_col] == mask_val]
    return {g: np.quantile(d[d.group == g]["rt"].values, QS) for g in GROUP}


def main():
    df = load_df().reset_index()
    sims = {title: ppc_rt(load_df(), tag, trace) for title, tag, trace in KEY}

    # ---- Figure 1: overall RT quantiles by group ----
    fig, axes = plt.subplots(1, len(KEY), figsize=(7.0, 2.9), sharey=True,
                             constrained_layout=True)
    for ax, (title, *_), in zip(axes, KEY):
        b = band(sims[title]); o = obs_q(df)
        for g, col in GROUP.items():
            med, lo, hi = b[g]
            ax.fill_between(QS, lo, hi, color=col, alpha=0.22, lw=0)
            ax.plot(QS, med, color=col, lw=1.3, zorder=3)
            ax.scatter(QS, o[g], s=26, facecolor="white", edgecolor=col, lw=1.3, zorder=4)
        ax.set_xticks(QS); ax.set_xlabel("RT quantile")
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("RT (s)")
    axes[0].annotate("Control", (0.05, 0.92), xycoords="axes fraction",
                     color=GROUP["control"], fontsize=8, fontweight="bold")
    axes[0].annotate("Dyscalc", (0.05, 0.82), xycoords="axes fraction",
                     color=GROUP["dyscalc"], fontsize=8, fontweight="bold")
    fig.suptitle("Posterior-predictive RT quantiles by group  "
                 "(points = data, line/band = model 94% PPC)", fontsize=9.5)
    figstyle.finish(fig, op.join(HERE, "notes", "figures", "key_models_rt_ppc.pdf"))

    # ---- Figure 2: RT quantiles split correct vs error (slow-error signature) ----
    fig2, axes2 = plt.subplots(1, len(KEY), figsize=(7.0, 2.9), sharey=True,
                               constrained_layout=True)
    for ax, (title, *_), in zip(axes2, KEY):
        for corr, ls, lab in [(True, "-", "correct"), (False, "--", "error")]:
            b = band(sims[title], "correct", corr); o = obs_q(df, "correct", corr)
            for g, col in GROUP.items():
                med, lo, hi = b[g]
                ax.plot(QS, med, color=col, lw=1.2, ls=ls, zorder=3)
                ax.scatter(QS, o[g], s=20, facecolor="white", edgecolor=col,
                           lw=1.1, zorder=4, marker=("o" if corr else "^"))
        ax.set_xticks(QS); ax.set_xlabel("RT quantile")
        ax.set_title(title, fontsize=9.5)
    axes2[0].set_ylabel("RT (s)")
    axes2[0].annotate("● correct  ▲ error", (0.04, 0.93), xycoords="axes fraction",
                      fontsize=7, color="0.3")
    fig2.suptitle("RT quantiles, correct vs error  (lines = model PPC; "
                  "solid=correct, dashed=error)", fontsize=9.5)
    figstyle.finish(fig2, op.join(HERE, "notes", "figures", "key_models_rt_ppc_accuracy.pdf"))


if __name__ == "__main__":
    main()
