"""Supplementary figure: individual-subject posterior estimates for the
trusted RDM (full drift) model, rank-ordered, both groups pooled onto one
x-axis per panel (a caterpillar plot). Complements supp_rdm_params.pdf
(the population-level summary): shows whether the group-level separation
there reflects a clean per-subject split or two heavily overlapping
individual distributions.

    python plot_supp_subject_params.py
    # reads notes/data/subject_params.tsv (per-subject summary, extracted
    # on the cluster from the rdm_max trace by collect_subject_params.py --
    # see notes/figures/PROVENANCE.md) and notes/data/overview.json (group
    # population means, for the reference lines)
    # writes notes/figures/supp_subject_params.{pdf,png}
"""
import json
import os
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import figstyle

figstyle.apply()
HERE = op.dirname(op.abspath(__file__))
GROUP = figstyle.GROUP

PARAMS = [
    ("perceptual_noise_sd", "Perceptual noise $\\sigma_p$"),
    ("memory_noise_sd", "Memory noise $\\sigma_m$"),
    ("a", "Threshold $a$"),
    ("t0", "Non-decision time $t_0$"),
    ("w_0", "Baseline drift $w_0$"),
    ("w_s", "Magnitude-sum drift $w_s$"),
]


def panel(ax, df, pk, label, raw):
    d = df[df["parameter"] == pk].sort_values("mean").reset_index(drop=True)
    n = len(d)
    x = np.arange(1, n + 1)
    for g in ("control", "dyscalc"):
        m = (d["group"] == g).values
        yerr = np.vstack([d["mean"].values[m] - d["lo"].values[m],
                          d["hi"].values[m] - d["mean"].values[m]])
        ax.errorbar(x[m], d["mean"].values[m], yerr=yerr, fmt="o", ms=3.4,
                    color=GROUP[g], ecolor=GROUP[g], elinewidth=0.8,
                    alpha=0.8, capsize=0, zorder=3)
    group_means = {g: raw[pk][g]["mean"] for g in ("control", "dyscalc")}
    for g in ("control", "dyscalc"):
        ax.axhline(group_means[g], color=GROUP[g], ls=":", lw=1.1, zorder=1)

    # Population-level summary (dumbbell from supp_rdm_params.pdf row 1),
    # appended past a gap so individual variability and the pooled estimate
    # read off the same axis -- the forest-plot "summary diamond" convention.
    ax.axvline(n + 1.5, color="0.85", lw=0.8, zorder=0)
    for i, g in enumerate(("control", "dyscalc")):
        xs = n + 3 + i
        r = raw[pk][g]
        ax.errorbar([xs], [r["mean"]], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                    fmt="D", ms=5.5, color=GROUP[g], ecolor=GROUP[g], elinewidth=1.6,
                    capsize=2.5, zorder=4)

    ax.set_title(label, fontsize=9.5)
    ax.set_xticks([(n + 1) / 2, n + 3.5])
    ax.set_xticklabels(["Subjects (rank-ordered)", "Population"], fontsize=7.5)
    ax.tick_params(axis="x", length=0)
    ax.get_xticklabels()[1].set_color("0.35")
    ax.margins(x=0.02)
    lo0 = min(0, d["lo"].min())
    hi0 = max(d["hi"].max(), raw[pk]["control"]["hi"], raw[pk]["dyscalc"]["hi"])
    ax.set_ylim(lo0, hi0 * 1.05)


def main():
    df = pd.read_csv(op.join(HERE, "notes/data/subject_params.tsv"), sep="\t")
    S = json.load(open(op.join(HERE, "notes/data/overview.json")))
    raw = S["rdm_max"]["raw"]

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), constrained_layout=True)
    for ax, (pk, label) in zip(axes.ravel(), PARAMS):
        panel(ax, df, pk, label, raw)
    axes[0, 0].set_ylabel("Value", fontsize=8.5)
    axes[1, 0].set_ylabel("Value", fontsize=8.5)

    handles = [plt.Line2D([0], [0], color=GROUP[g], lw=0, marker="o", ms=6,
                          markeredgecolor="white", markeredgewidth=0.8)
              for g in ("control", "dyscalc")]
    fig.legend(handles, ["Control", "Dyscalc"], loc="outside upper right",
              frameon=False, fontsize=8.5, handlelength=1.0)

    fig.suptitle("RDM (full drift) — individual-subject parameter estimates",
                 fontsize=10.5, x=0.42, y=1.06)
    fig.text(0.5, -0.02,
             "Each point = one subject's posterior mean ± 94% HDI (natural "
             "units), ranked within each panel by point estimate; groups "
             "pooled onto one x-axis. Dotted lines and the diamonds past "
             "the gap: population mean ± 94% HDI per group (same values as "
             "supp_rdm_params.pdf row 1).",
             ha="center", va="top", fontsize=7.5, style="italic", color="0.35")

    sns.despine(fig=fig, offset=3, trim=False)
    for ax in axes.ravel():
        ax.tick_params(bottom=False)

    out = op.join(HERE, "notes/figures/supp_subject_params.pdf")
    os.makedirs(op.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out[:-4] + ".png", dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
