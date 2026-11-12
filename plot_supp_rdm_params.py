"""Supplementary figure: full posterior parameter summary for the trusted
model -- RDM (full drift), the "rdm_max" fit with w0/ws freed (see
CLAUDE.md's headline results). Top row: population-level group means with
94% HDI, in natural units (what CONCLUSIONS.md quotes). Middle row: the
dyscalc-control contrast itself with its 94% HDI (link scale -- the actual
quantity the hypothesis test is run on; natural-unit group means don't
subtract validly because the softplus transform is nonlinear and the two
group posteriors are correlated). Bottom row: between-subject SD (tau) per
group -- how heterogeneous each group is, not just where its average sits.

    python plot_supp_rdm_params.py
    # reads notes/data/overview.json (the "rdm_max" entry)
    # writes notes/figures/supp_rdm_params.{pdf,png}
"""
import json
import os
import os.path as op

import matplotlib.pyplot as plt
import seaborn as sns

import figstyle

figstyle.apply()
HERE = op.dirname(op.abspath(__file__))
GROUP = figstyle.GROUP

PARAMS = [
    ("perceptual_noise_sd", "Perceptual\nnoise $\\sigma_p$"),
    ("memory_noise_sd", "Memory\nnoise $\\sigma_m$"),
    ("a", "Threshold\n$a$"),
    ("t0", "Non-decision\ntime $t_0$"),
    ("w_0", "Baseline\ndrift $w_0$"),
    ("w_s", "Magnitude-sum\ndrift $w_s$"),
]
OFFSET = 0.18
YLIM = (-0.42, 0.42)


def _fmt_p(p):
    if p >= 0.995:
        return "P>0.99"
    if p <= 0.005:
        return "P<0.01"
    return f"P={p:.2f}"


def _zero_anchored_xlim(ax, hi):
    """All these parameters are non-negative (softplus) or bounded well away
    from 0 (w_s) -- anchoring the axis at 0 makes magnitudes and the
    zero-effect reference for w_s legible, not just the CI span."""
    ax.set_xlim(0, hi * 1.15)
    ax.axvline(0, color="0.75", lw=0.8, zorder=1)


def panel_raw(ax, d, pk):
    """Population mean +- 94% HDI per group, natural units."""
    r = d["raw"][pk]
    for g, y in (("control", OFFSET), ("dyscalc", -OFFSET)):
        v, col = r[g], GROUP[g]
        ax.plot([v["lo"], v["hi"]], [y, y], color=col, lw=2.8,
                solid_capstyle="round", zorder=2)
        ax.scatter([v["mean"]], [y], s=44, color=col, edgecolor="white",
                   lw=1.1, zorder=3)
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    _zero_anchored_xlim(ax, max(r["control"]["hi"], r["dyscalc"]["hi"]))


def panel_contrast(ax, contrast):
    """The dyscalc - control contrast with its own 94% HDI, link scale --
    the CI the P(Δ>0) headline number is actually computed from."""
    credible = contrast["lo"] > 0 or contrast["hi"] < 0
    col = GROUP["dyscalc"] if (credible and contrast["mean"] > 0) else \
        (GROUP["control"] if credible else "0.55")
    ax.plot([contrast["lo"], contrast["hi"]], [0, 0], color=col, lw=2.8,
            solid_capstyle="round", zorder=2)
    ax.scatter([contrast["mean"]], [0], s=44, color=col, edgecolor="white",
               lw=1.1, zorder=3)
    ax.axvline(0, ls=":", color="0.5", lw=0.9, zorder=1)
    ax.annotate(_fmt_p(contrast["Pgt0"]), (0.5, 0.86), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=7.8, color=col,
                fontweight="bold" if credible else "normal")
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    ax.margins(x=0.35)


def panel_tau(ax, d, pk):
    """Between-subject SD (tau) per group -- link scale, heterogeneity."""
    gs = d["group_sds"][pk]
    for g, y in (("control", OFFSET), ("dyscalc", -OFFSET)):
        v, col = gs[g], GROUP[g]
        ax.plot([v["lo"], v["hi"]], [y, y], color=col, lw=2.2,
                solid_capstyle="round", zorder=2, alpha=0.8)
        ax.scatter([v["mean"]], [y], s=30, color=col, edgecolor="white",
                   lw=0.9, zorder=3, alpha=0.8)
    p = gs["contrast"]["Pgt0"]
    credible = gs["contrast"]["lo"] > 0 or gs["contrast"]["hi"] < 0
    ax.annotate(_fmt_p(p), (0.5, 0.86), xycoords="axes fraction", ha="center",
                va="bottom", fontsize=7.2, color="0.3" if credible else "0.55")
    ax.set_ylim(*YLIM)
    ax.set_yticks([])
    _zero_anchored_xlim(ax, max(gs["control"]["hi"], gs["dyscalc"]["hi"]))


def main():
    S = json.load(open(op.join(HERE, "notes/data/overview.json")))
    d = S["rdm_max"]

    fig, axes = plt.subplots(3, len(PARAMS), figsize=(10.6, 6.4),
                              constrained_layout=True)
    for j, (pk, label) in enumerate(PARAMS):
        panel_raw(axes[0, j], d, pk)
        panel_contrast(axes[1, j], d["contrasts"][pk])
        panel_tau(axes[2, j], d, pk)
        axes[0, j].set_title(label, fontsize=9)
        axes[2, j].set_xlabel("Value", fontsize=7.5)

    axes[0, 0].set_ylabel("Population\nmean\n(94% HDI)", fontsize=8, labelpad=8)
    axes[1, 0].set_ylabel("Δ dyscalc −\ncontrol\n(94% HDI,\nlink scale)",
                          fontsize=8, labelpad=8)
    axes[2, 0].set_ylabel("Between-subject\nSD τ\n(94% HDI, link scale)",
                          fontsize=8, labelpad=8)
    handles = [plt.Line2D([0], [0], color=GROUP[g], lw=3, marker="o", ms=6,
                          markeredgecolor="white", markeredgewidth=1)
              for g in ("control", "dyscalc")]
    fig.legend(handles, ["Control", "Dyscalc"], loc="outside upper right",
              ncol=1, frameon=False, fontsize=8.5, handlelength=1.3)

    fig.suptitle("RDM (full drift) — population parameter estimates by group",
                 fontsize=10.5, x=0.42, y=1.05)
    fig.text(0.5, -0.02,
             "Row 1 values in natural units; rows 2-3 on the link/"
             "unconstrained scale (not comparable across parameters, or to "
             "row 1). Bold P = 94% HDI excludes 0.",
             ha="center", va="top", fontsize=7.5, style="italic", color="0.35")

    sns.despine(fig=fig, offset=3, trim=False)
    for ax in axes.ravel():
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    out = op.join(HERE, "notes/figures/supp_rdm_params.pdf")
    os.makedirs(op.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out[:-4] + ".png", dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
