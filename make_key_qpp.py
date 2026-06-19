"""Quantile-probability plots (Ratcliff / HSSM idiom), conditioned on SIGNED
difficulty so both difficulty AND order are visible.

Conditions = bins of signed log(n2/n1): from "n1 much larger" (left, P(chose
n2)→0) through the hard midpoint (ratio≈1, P≈0.5) to "n2 much larger" (right,
P→1). For each condition we plot the {.1,.3,.5,.7,.9} RT quantiles of BOTH
responses at x = P(chose n2): chose-n2 = filled, chose-n1 = open. Model =
translucent posterior-predictive cloud (per draw); data = bold points.

Grid: rows = model (DDM, RDM), cols = group (control, dyscalc).

  ~/mambaforge/envs/bauer/bin/python make_key_qpp.py
"""
import os.path as op
import warnings

import arviz as az
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import figstyle
import fit_ddm

warnings.filterwarnings("ignore")
figstyle.apply()
HERE = op.dirname(op.abspath(__file__))
QS = [0.1, 0.3, 0.5, 0.7, 0.9]
QCOL = sns.color_palette("mako", as_cmap=True)(np.linspace(0.15, 0.85, 5))
N_BINS = 7
N_PPC = 50
KEY = [("DDM", "ddm_group", "results/model-ddm_group_long_trace.nc"),
       ("RDM", "rdm_group", "results/model-rdm_group_long_trace.nc")]


def load_df():
    df = pd.read_csv(op.join(HERE, "data", "magjudge_behavior_DNumRisk.csv"))
    df = df[df["rt"] >= 0.20].copy()
    df["choice"] = df["chose_n2"].astype(bool)
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    df["L"] = np.log(df["n2"] / df["n1"])
    return df.set_index(["subject", "run", "trial_nr"])


def cond_quantiles(d, rt_col, ch_col, edges, by_sample=False):
    """Per (group, signed-difficulty bin[, draw]): x=P(chose n2) and the RT
    quantiles of each response, placed at x."""
    d = d.copy()
    d["bin"] = pd.cut(d["L"], edges, include_lowest=True)
    keys = ["group", "bin"] + (["ppc_sample"] if by_sample else [])
    rows = []
    for k, sub in d.groupby(keys, observed=True):
        kd = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        if len(sub) < 8:
            continue
        p = sub[ch_col].mean()                      # P(chose n2) for this bin
        for chose_n2 in (True, False):
            rts = sub.loc[sub[ch_col] == chose_n2, rt_col]
            if len(rts) < 5:
                continue
            # Ratcliff convention: each response at x = ITS OWN proportion ->
            # chose-n2 on the right wing (x=p), chose-n1 on the left (x=1-p).
            x = p if chose_n2 else 1 - p
            for qi, q in enumerate(QS):
                rows.append({**kd, "qi": qi, "x": x, "chose_n2": chose_n2,
                             "rt": rts.quantile(q)})
    return pd.DataFrame(rows)


def ppc_choices(df, tag, trace):
    m = fit_ddm.build_model(tag, df)
    m.build_estimation_model(df, hierarchical=True)
    idata = az.from_netcdf(op.join(HERE, trace))
    sim = m.ppc(df, idata, n_posterior_samples=N_PPC, progressbar=False)
    sim = sim.join(df[["group", "n1", "n2", "L"]])
    sim = sim[sim["simulated_rt"] >= 0.20].copy()
    sim["chose_n2"] = sim["simulated_choice"].astype(bool)
    return sim


def main():
    df = load_df()
    edges = np.unique(np.quantile(df["L"], np.linspace(0, 1, N_BINS + 1)))
    obs = df.reset_index()
    data_q = cond_quantiles(obs, "rt", "choice", edges)
    ymax = float(df["rt"].quantile(0.985)) * 1.1
    groups = ["control", "dyscalc"]

    fig, axes = plt.subplots(len(KEY), 2, figsize=(7.4, 5.4), sharex=True,
                             sharey=True, constrained_layout=True)
    for r, (title, tag, trace) in enumerate(KEY):
        sim = ppc_choices(df, tag, trace)
        model_q = cond_quantiles(sim, "simulated_rt", "chose_n2", edges, by_sample=True)
        for c, g in enumerate(groups):
            ax = axes[r, c]
            mg = model_q[model_q.group == g]; dg = data_q[data_q.group == g]
            for qi in range(5):
                for chose in (True, False):
                    # model: median across draws per bin -> a smooth fan line
                    mm = mg[(mg.qi == qi) & (mg.chose_n2 == chose)]
                    if len(mm):
                        agg = (mm.groupby("bin", observed=True)
                                 .agg(x=("x", "median"), rt=("rt", "median"))
                                 .sort_values("x"))
                        ax.plot(agg.x, agg.rt, color=QCOL[qi], lw=1.1,
                                ls="-" if chose else ":", alpha=0.75, zorder=2)
                    # data: points (filled = chose n2, open = chose n1)
                    dd = dg[(dg.qi == qi) & (dg.chose_n2 == chose)].sort_values("x")
                    ax.scatter(dd.x, dd.rt, s=18,
                               facecolor=QCOL[qi] if chose else "white",
                               edgecolor=QCOL[qi], lw=0.9, zorder=3)
            ax.set_xlim(0, 1); ax.set_ylim(0, ymax); ax.set_xticks([0, 0.5, 1])
            if r == 0:
                ax.set_title(g.capitalize(), fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"{title}\nRT quantile (s)", fontsize=9)
            if r == len(KEY) - 1:
                ax.set_xlabel("P(chose n2)")
    axes[0, 0].annotate("● chose n2 (right) ○ chose n1 (left)\n"
                        "lines = model PPC · points = data",
                        (0.5, 0.97), xycoords="axes fraction", va="top", ha="center",
                        fontsize=6.3, color="0.35")
    fig.suptitle("Quantile-probability by signed difficulty "
                 "(each response at its own proportion)", fontsize=9.5)
    figstyle.finish(fig, op.join(HERE, "notes", "figures", "key_models_qpp.pdf"))


if __name__ == "__main__":
    main()
