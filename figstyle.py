"""Shared publication house style (vision-science / scientific-figures skill).

Import and call `apply()` at the top of any figure script:

    import figstyle; figstyle.apply()

Then finish each figure with `figstyle.finish(fig)` (despine offset+trim, save
PDF). Group palette in `GROUP`.
"""
import matplotlib as mpl
import seaborn as sns

# Muted, deliberate categorical palette (blue control, red dyscalc).
GROUP = {"control": "#3B5BA5", "dyscalc": "#C44E52"}
NEUTRAL = "#7F7F7F"


def apply():
    mpl.rcParams.update({
        "font.family": "Helvetica",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Arial"],
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "mathtext.fontset": "stixsans",
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelpad": 4,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.2, "lines.markersize": 4, "patch.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    sns.set_context("paper")


def panel_letter(ax, letter):
    ax.text(-0.18, 1.05, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="right")


def finish(fig, out, offset=5, trim=True):
    import os, os.path as op
    sns.despine(fig=fig, offset=offset, trim=trim)
    os.makedirs(op.dirname(op.abspath(out)), exist_ok=True)
    fig.savefig(out)                      # PDF; rcParams handle fonts/dpi
    if out.endswith(".pdf"):              # PNG sibling for quick on-screen viewing
        fig.savefig(out[:-4] + ".png", dpi=200)
    print(f"Saved {out}")
