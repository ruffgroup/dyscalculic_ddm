"""Supplementary figure: cross-model recovery check for the pure-PCM
dyscalc<control perceptual-noise paradox. Christian's hypothesis: a
choice-only model (no threshold parameter) can't tell "less noise" apart
from "more noise, compensated by a higher threshold" -- so the PCM's
reversed contrast may be a modeling artifact, not a real perceptual
advantage in dyscalculics. Tested here mechanically (not by correlating
noisy subject-level point estimates): simulate choices from the REAL
rdm_max posterior (noise AND threshold genuinely higher for dyscalc, both
baked into each draw), drop the RTs, refit the pure PCM to the simulated
choices, and see whether the paradox reappears.

    python plot_supp_recover_cross_pcm.py
    # reads results/recover_cross_pcm/rep*.csv if present (recover_cross_pcm.py
    # output, run on the cluster -- see notes/figures/PROVENANCE.md), else the
    # committed notes/data/recover_cross_pcm.tsv summary; plus
    # notes/data/overview.json (the real "choice" tag contrast, for reference)
    # writes notes/figures/supp_recover_cross_pcm.{pdf,png}
"""
import glob
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
TSV = op.join(HERE, "notes/data/recover_cross_pcm.tsv")
TRUTH_COLOR = "#2E7D6B"     # matches RDM_COLOR in supp_ddm_vs_rdm.py
PCM_COLOR = "#C44E52"       # dyscalc red -- this is where the paradox lives
FLIP_COLOR = "#C44E52"
NOFLIP_COLOR = "#7F7F7F"


def load():
    """Prefer the per-rep CSVs when the cluster output is present locally (and
    refresh the committed summary from them); fall back to the committed TSV so
    the figure regenerates from git alone -- `results/` is gitignored."""
    files = sorted(glob.glob(op.join(HERE, "results", "recover_cross_pcm", "rep*.csv")))
    if files:
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True).sort_values("rep")
        df.to_csv(TSV, sep="\t", index=False)
        return df
    return pd.read_csv(TSV, sep="\t")


def main():
    df = load()
    n_raw = len(df)
    df = df[df["max_rhat"] <= 1.05].sort_values("true_perceptual_noise_sd").reset_index(drop=True)

    S = json.load(open(op.join(HERE, "notes/data/overview.json")))
    real_pcm = S["choice"]["contrasts"]["perceptual_noise_sd"]["mean"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # --- Panel A: slope graph, TRUE (simulated truth) -> REFIT (PCM) ---
    # n=997 reps -> draw the full distribution as violins (exact) and a
    # random subsample of individual connecting lines on top (legible
    # "flip" storytelling; the annotated flip fraction is from the FULL
    # sample, not just what's drawn).
    x = [0, 1]
    n_flip = (df["rec_perceptual_noise_sd"] < 0).sum()
    parts = axA.violinplot([df["true_perceptual_noise_sd"], df["rec_perceptual_noise_sd"]],
                           positions=x, widths=0.5, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("0.75"); body.set_edgecolor("none"); body.set_alpha(0.5)
        body.set_zorder(1)
    rng = np.random.default_rng(0)
    show = df.iloc[rng.choice(len(df), size=min(150, len(df)), replace=False)]
    for _, r in show.iterrows():
        flipped = r["rec_perceptual_noise_sd"] < 0
        col = FLIP_COLOR if flipped else NOFLIP_COLOR
        axA.plot(x, [r["true_perceptual_noise_sd"], r["rec_perceptual_noise_sd"]],
                 color=col, lw=0.8, alpha=0.35, zorder=2,
                 marker="o", ms=2.6, mfc=col, mec="none")
    axA.axhline(0, ls=":", color="0.5", lw=0.9, zorder=1)
    axA.axhline(real_pcm, ls="--", color=PCM_COLOR, lw=1.3, zorder=1)
    axA.annotate("Real-data PCM", (0.02, real_pcm),
                 xycoords=("axes fraction", "data"), xytext=(0, 3),
                 textcoords="offset points", fontsize=7.3, color=PCM_COLOR,
                 va="bottom", ha="left")
    axA.annotate(f"{n_flip}/{len(df)} reps flip sign ({n_flip/len(df):.0%})\n"
                 f"(150 of {len(df)} shown)", (0.5, 0.03),
                 xycoords="axes fraction", ha="center", fontsize=8, color="0.3")
    axA.set_xticks(x)
    axA.set_xticklabels(["True\n(simulated from\nthe RDM)", "Refit\n(choice-only\nPCM)"],
                        fontsize=8)
    axA.set_ylabel("Perceptual-noise contrast\nΔ (dyscalc − control), link scale")
    axA.margins(x=0.15)

    # --- Panel B: distribution of the refit contrast vs the real PCM value ---
    axB.hist(df["rec_perceptual_noise_sd"], bins=30, color=NOFLIP_COLOR, alpha=0.75,
             edgecolor="white", lw=0.4)
    axB.axvline(0, ls=":", color="0.5", lw=0.9)
    axB.axvline(real_pcm, ls="--", color=PCM_COLOR, lw=1.3)
    axB.axvline(df["rec_perceptual_noise_sd"].mean(), color=NOFLIP_COLOR, lw=1.3)
    # unicode minus, to match the tick labels
    fmt = lambda v: f"{v:+.2f}".replace("-", "\u2212")
    axB.text(0.02, 0.98, f"Recovered mean = {fmt(df['rec_perceptual_noise_sd'].mean())}",
             transform=axB.transAxes, fontsize=7.3, color="0.25", va="top", ha="left")
    axB.text(0.02, 0.90, f"Real-data PCM = {fmt(real_pcm)}", transform=axB.transAxes,
             fontsize=7.3, color=PCM_COLOR, va="top", ha="left")
    axB.set_xlabel("Refit PCM perceptual-noise contrast\n(link scale)")
    axB.set_ylabel("Reps")

    figstyle.panel_letter(axA, "a")
    figstyle.panel_letter(axB, "b")
    # Deliberately no figure-level prose: the title is one line and everything
    # else that used to sit under the panels is in notes/figures/CAPTIONS.md.
    fig.suptitle("The choice-only model reverses the noise difference it was "
                 "simulated from", fontsize=10, y=1.03)

    sns.despine(fig=fig, offset=5, trim=True)
    out = op.join(HERE, "notes/figures/supp_recover_cross_pcm.pdf")
    os.makedirs(op.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out[:-4] + ".png", dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
