"""Supplementary figure: the RDM captures the overall-magnitude effect on RT;
the DDM does not -- even though both models reproduce the standard
chronometric (RT-by-difficulty) effect. Reconstruction of a figure whose
original plotting script was lost (never committed -- see
notes/figures/PROVENANCE.md for the incident), since extended with a
chronometric-function panel. Rebuilt from committed summary data only;
PROVENANCE.md in notes/figures/ documents exactly what is exact vs.
approximated in the reconstructed part.

    python plot_supp_ddm_vs_rdm.py
    # reads notes/data/overview.{json,npz}
    # writes notes/figures/supp_ddm_vs_rdm.{pdf,png}
"""
import json
import os
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import figstyle

figstyle.apply()
HERE = op.dirname(op.abspath(__file__))

# Model colors deliberately NOT figstyle.GROUP's blue/red -- those mean
# control/dyscalc everywhere else in this paper's figures. This figure pools
# both groups and contrasts two MODELS instead, so it gets its own palette:
# grey for the model that fails (DDM), teal for the one that wins (RDM).
DATA_COLOR = "#222222"
DDM_COLOR = figstyle.NEUTRAL   # "#7F7F7F"
RDM_COLOR = "#2E7D6B"


def _pool(npz, prefix, suffix):
    """Equal-weight mean of the two (equal-N=33) groups' committed summaries."""
    c = npz[f"{prefix}__control__{suffix}"]
    d = npz[f"{prefix}__dyscalc__{suffix}"]
    return (c + d) / 2.0


def _rt_panel(ax, npz, key, xlabel, label_offsets, subtitle):
    x = npz[f"obsrt__{key}__control__x"]
    data_y = _pool(npz, f"obsrt__{key}", "y")

    ax.scatter(x, data_y, s=26, facecolor="white", edgecolor=DATA_COLOR,
               lw=1.2, zorder=4)

    for tag, color, label in [("ddm", DDM_COLOR, "DDM"), ("rdm", RDM_COLOR, "RDM")]:
        med = _pool(npz, f"{tag}__{key}", "med")
        lo = _pool(npz, f"{tag}__{key}", "lo")
        hi = _pool(npz, f"{tag}__{key}", "hi")
        ax.fill_between(x, lo, hi, color=color, alpha=0.22, lw=0, zorder=2)
        ax.plot(x, med, color=color, lw=1.4, zorder=3)
        ax.annotate(label, (x[-1], med[-1]), xytext=label_offsets[tag],
                    textcoords="offset points", va="center", ha="left",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median RT (s)")
    ax.margins(x=0.20)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, fontsize=7.3,
            color="0.35", ha="center", va="bottom", style="italic")


def panel_difficulty(ax, npz):
    _rt_panel(ax, npz, "difficulty", "|log(n2 / n1)| (difficulty)",
              label_offsets={"ddm": (4, 7), "rdm": (4, -9)},
              subtitle="both models track the chronometric effect")


def panel_magnitude(ax, npz):
    _rt_panel(ax, npz, "magnitude", "n1 + n2 (overall magnitude)",
              label_offsets={"ddm": (4, 0), "rdm": (4, 0)},
              subtitle="only the RDM captures the magnitude effect")
    ax.set_xticks([10, 20, 30, 40])


def panel_loo(ax, S):
    rows = [("rdm", RDM_COLOR), ("ddm", DDM_COLOR)]
    ref = S["rdm"]["elpd_loo"]
    ys = np.arange(len(rows))[::-1]
    for y, (tag, color) in zip(ys, rows):
        d = S[tag]
        delta = d["elpd_loo"] - ref
        ax.errorbar(delta, y, xerr=d["elpd_se"], fmt="o", ms=7, color=color,
                     capsize=0, lw=1.2)
        bad_frac = d["bad_k"] / d["n_obs"]
        flag = f", Pareto-k unreliable ({d['bad_k']}/{d['n_obs']} trials)" \
            if bad_frac > 0.005 else ""
        ax.annotate(f"{d['pretty']}: Δelpd {delta:+.0f}{flag}",
                    (delta, y), xytext=(0, 12), textcoords="offset points",
                    va="bottom", ha="center", fontsize=8, color="0.25")
    ax.axvline(0, ls=":", color="0.5", lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.tick_params(left=False)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.margins(x=0.35)
    ax.set_xlabel("Δ elpd_loo vs RDM (higher = better)")


def main():
    npz = np.load(op.join(HERE, "notes/data/overview.npz"), allow_pickle=True)
    S = json.load(open(op.join(HERE, "notes/data/overview.json")))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(9.8, 3.0),
                                         constrained_layout=True)
    panel_difficulty(axA, npz)
    panel_magnitude(axB, npz)
    panel_loo(axC, S)

    ylo = min(axA.get_ylim()[0], axB.get_ylim()[0])
    yhi = max(axA.get_ylim()[1], axB.get_ylim()[1])
    axA.set_ylim(ylo, yhi)
    axB.set_ylim(ylo, yhi)

    figstyle.panel_letter(axA, "a")
    figstyle.panel_letter(axB, "b")
    figstyle.panel_letter(axC, "c")
    fig.suptitle("Both models reproduce the chronometric effect; only the "
                 "racing-diffusion model reproduces the magnitude effect",
                 fontsize=10, y=1.08)

    # Not figstyle.finish(): its shared sns.despine(left=False) call would
    # force panel c's left spine back on (seaborn sets visibility
    # unconditionally for every side), undoing panel_loo()'s intentional
    # removal of that spine (the y-axis carries no information there).
    sns.despine(fig=fig, offset=5, trim=True)
    axC.spines["left"].set_visible(False)
    out = op.join(HERE, "notes/figures/supp_ddm_vs_rdm.pdf")
    os.makedirs(op.dirname(out), exist_ok=True)
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
