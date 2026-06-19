"""Does the compression (n1) effect need perceptual or memory noise?
P(chose n2) vs n1, per group: data points vs the perceptual-only and
memory-only RDM posterior-predictives. If memory-only reproduces the steeper
dyscalc slope where perceptual-only underfits it, the compression demands
memory noise.
"""
import os.path as op
import arviz as az
import numpy as np
import pandas as pd
import figstyle


def band(cache):
    ppc = pd.read_parquet(cache)
    out = {}
    for g in ["control", "dyscalc"]:
        per = (ppc[ppc.group == g].groupby(["n1", "ppc_sample"])["simulated_choice"]
               .mean().reset_index())
        agg = per.groupby("n1")["simulated_choice"].agg(
            m="mean", lo=lambda x: az.hdi(x.values, 0.94)[0],
            hi=lambda x: az.hdi(x.values, 0.94)[1])
        out[g] = agg
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--perc-tag", default="rdm_ponly")
    ap.add_argument("--mem-tag", default="rdm_monly")
    ap.add_argument("--arch", default="RDM")
    ap.add_argument("--out", default="notes/figures/compression_perc_vs_mem.pdf")
    args = ap.parse_args()
    figstyle.apply()
    import matplotlib.pyplot as plt
    df = pd.read_csv("data/magjudge_behavior_DNumRisk.csv")
    df = df[df["rt"] >= 0.20]
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    perc = band(f"results/ppc_{args.perc_tag}.parquet")
    mem = band(f"results/ppc_{args.mem_tag}.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    for ax, (mname, mb, ls) in zip(
            axes, [(f"{args.arch} σ-perceptual-only", perc, "-"),
                   (f"{args.arch} σ-memory-only", mem, "-")]):
        for g in ["control", "dyscalc"]:
            o = df[df.group == g].groupby("n1")["chose_n2"].mean()
            ax.scatter(o.index, o.values, color=figstyle.GROUP[g], s=32, zorder=3)
            b = mb[g]
            ax.plot(b.index, b["m"], color=figstyle.GROUP[g], lw=1.8, ls=ls)
            ax.fill_between(b.index, b.lo, b.hi, color=figstyle.GROUP[g], alpha=.18)
        ax.axhline(0.5, ls=":", c="0.6", lw=0.7)
        ax.set_xlabel("n1 (reference)")
        ax.set_title(mname, fontsize=9)
    axes[0].set_ylabel("P(chose n2)")
    axes[0].annotate("Points = data\nBand = model", xy=(0.04, 0.96),
                     xycoords="axes fraction", fontsize=6.5, color="0.35", va="top")
    # annotate dyscalc steepness
    axes[1].annotate("Does dyscalc (red)\nslope steepen?", xy=(0.5, 0.06),
                     xycoords="axes fraction", fontsize=7, color=figstyle.GROUP["dyscalc"])
    fig.suptitle("Which noise reproduces the dyscalc compression (n1) effect?",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figstyle.finish(fig, args.out)


if __name__ == "__main__":
    main()
