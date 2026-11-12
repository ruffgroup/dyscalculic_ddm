"""Supplementary figure: raw median RT as a function of log(n2/n1), split by
the reference (first) number n1 and by group. Model-free; plots the
behavioural data directly.

Binning: log(n2/n1) is cut into 8 global quantile bins (pd.qcut over all
trials, both groups, all n1), so every line shares the same x positions.
x = mean log(n2/n1) of the trials in each bin; y = median RT pooled over
all trials of that (group, n1, bin) cell (not a median of subject medians).

    python plot_supp_rt_by_n1_group.py
    # reads data/magjudge_behavior_DNumRisk.csv (gitignored, human-subjects)
    # writes notes/figures/supp_rt_by_n1_group.{pdf,png}

Reconstruction note: the original script for this figure (2026-08-17) was
never committed. This one reproduces the original figure's values; see
notes/figures/PROVENANCE.md.
"""
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import figstyle

figstyle.apply()
HERE = op.dirname(op.abspath(__file__))
DATA = op.join(HERE, "data", "magjudge_behavior_DNumRisk.csv")
OUT = op.join(HERE, "notes", "figures", "supp_rt_by_n1_group.pdf")
N_BINS = 8
LS = {"control": "-", "dyscalc": "--"}
LABEL = {"control": "Control", "dyscalc": "Dyscalc"}


def summarize(df):
    df = df.copy()
    df["lr"] = df["log(n2/n1)"]
    df["bin"] = pd.qcut(df["lr"], N_BINS)
    x = df.groupby("bin", observed=True)["lr"].mean().rename("x")
    s = (df.groupby(["group", "n1", "bin"], observed=True)["rt"].median()
         .rename("median_rt").reset_index()
         .merge(x, left_on="bin", right_index=True))
    return s


def main():
    s = summarize(pd.read_csv(DATA))
    n1s = sorted(s["n1"].unique())
    pal = dict(zip(n1s, plt.cm.viridis(np.linspace(0, 1, len(n1s)))))

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    for (g, n1), sub in s.groupby(["group", "n1"]):
        sub = sub.sort_values("x")
        ax.plot(sub["x"], sub["median_rt"], ls=LS[g], color=pal[n1],
                marker="o", ms=3.5)
    ax.axvline(0, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("log(n2 / n1)")
    ax.set_ylabel("Median RT (s)")
    ax.set_title("RT by reference n1 and group")

    leg1 = ax.legend([Line2D([], [], color=pal[n], lw=2) for n in n1s],
                     [f"n1 = {n}" for n in n1s], title="Reference n1",
                     loc="upper left")
    ax.add_artist(leg1)
    ax.legend([Line2D([], [], color="0.2", ls=LS[g], lw=1.5) for g in LS],
              [LABEL[g] for g in LS], loc="lower right")
    figstyle.finish(fig, OUT)


if __name__ == "__main__":
    main()
