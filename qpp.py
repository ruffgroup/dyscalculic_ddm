"""Quantile-probability plot (QPP), SSM/HSSM idiom: quantiles color-coded,
model shown as a translucent posterior-predictive CLOUD (one point per draw),
data as bold markers. House style.

For each difficulty condition (|log(n2/n1)| bins) we place the {.1,.3,.5,.7,.9}
RT quantiles at the response proportion — correct responses at x = P(correct)
(right wing), errors at x = 1 - P(correct) (left wing). The model's predictive
spread is the per-draw scatter; data are the solid points.

    ~/mambaforge/envs/bauer/bin/python qpp.py --model rdm_group \
        --out notes/figures/qpp_rdm.pdf
"""
import argparse
import os.path as op

import numpy as np
import pandas as pd
import seaborn as sns

import figstyle

QUANTILES = [0.1, 0.3, 0.5, 0.7, 0.9]
N_BINS = 5
QCOLORS = sns.color_palette("mako", as_cmap=True)(np.linspace(0.15, 0.85, 5))


def load_observed():
    df = pd.read_csv("data/magjudge_behavior_DNumRisk.csv")
    df = df[df["rt"] >= 0.20].copy()
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    df["chose_n2"] = df["chose_n2"].astype(bool)
    df["correct"] = df["chose_n2"] == (df["n2"] > df["n1"])
    df["absL"] = np.abs(np.log(df["n2"] / df["n1"]))
    return df


def per_condition(d, rt_col, corr_col, edges, by_sample=False):
    """RT quantiles + response proportion per (group, difficulty bin[, draw])."""
    d = d.copy()
    d["bin"] = pd.cut(d["absL"], edges, include_lowest=True)
    keys = ["group", "bin"] + (["ppc_sample"] if by_sample else [])
    rows = []
    for k, sub in d.groupby(keys, observed=True):
        kd = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        if len(sub) < 5:
            continue
        p = sub[corr_col].mean()
        for is_corr, x in [(True, p), (False, 1 - p)]:
            rts = sub.loc[sub[corr_col] == is_corr, rt_col]
            if len(rts) < 5:
                continue
            for qi, q in enumerate(QUANTILES):
                rows.append({**kd, "qi": qi, "x": x, "rt": rts.quantile(q)})
    return pd.DataFrame(rows)


def main(model_label, out):
    obs = load_observed()
    edges = np.unique(np.quantile(obs["absL"], np.linspace(0, 1, N_BINS + 1)))
    data_q = per_condition(obs, "rt", "correct", edges)
    # y-window from the data (model can have a floored-drift slow tail we clip)
    ymax = obs["rt"].quantile(0.99) * 1.15

    cache = f"results/ppc_{model_label}.parquet"
    if not op.exists(cache):
        raise SystemExit(f"missing {cache}; run ppc_choice.py --model {model_label} first")
    ppc = pd.read_parquet(cache)
    ppc["correct"] = ppc["simulated_choice"].astype(bool) == (ppc["n2"] > ppc["n1"])
    ppc["absL"] = np.abs(np.log(ppc["n2"] / ppc["n1"]))
    model_q = per_condition(ppc, "simulated_rt", "correct", edges, by_sample=True)

    figstyle.apply()
    import matplotlib.pyplot as plt
    groups = ["control", "dyscalc"]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.3), sharex=True, sharey=True)
    for i, (ax, g) in enumerate(zip(axes, groups)):
        m = model_q[model_q.group == g]
        d = data_q[data_q.group == g]
        for qi in range(5):
            mm = m[m.qi == qi]
            ax.scatter(mm.x, mm.rt, s=4, color=QCOLORS[qi], alpha=0.05,
                       lw=0, zorder=1)                      # predictive cloud
            dd = d[d.qi == qi].sort_values("x")
            ax.scatter(dd.x, dd.rt, s=15, color=QCOLORS[qi], edgecolor="k",
                       lw=0.4, alpha=0.85, zorder=3)        # data (small)
            if i == 1 and len(dd):                          # label quantiles
                r = dd.iloc[-1]
                ax.annotate(f".{int(QUANTILES[qi]*10)}", xy=(r.x, r.rt),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=6.5, color=QCOLORS[qi], va="center")
        ax.set_xlabel("Response proportion"); ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 0.5, 1.0])
        figstyle.panel_letter(ax, "ab"[i])
        ax.annotate(g.capitalize(), xy=(0.5, 0.97), xycoords="axes fraction",
                    ha="center", va="top", fontsize=9, color="0.2")
    axes[0].set_ylabel("RT quantile (s)")
    axes[0].annotate("Cloud = posterior\npredictive draws\nPoints = data",
                     xy=(0.04, 0.04), xycoords="axes fraction", fontsize=6.5,
                     color="0.3", va="bottom", ha="left")
    figstyle.finish(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ddm_group")
    p.add_argument("--out", default="notes/figures/qpp.pdf")
    args = p.parse_args()
    main(args.model, args.out)
