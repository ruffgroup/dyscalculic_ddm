"""Supplementary figure: the RDM threshold difference *is* what the choice-only
PCM reads as a perceptual-noise difference — shown by intervention, not
correlation.

`supp_recover_cross_pcm.pdf` established that a choice-only PCM refit to data
simulated from the real rdm_max posterior reproduces the paradoxical
dyscalc<control noise contrast. That is observational: every simulated world had
both a higher dyscalc threshold and higher dyscalc noise. Here each group
contrast is rescaled in the generating draw *one at a time*
(`recover_threshold_leak.py`), so the question "does the threshold end up in the
noise parameter?" gets a causal answer.

    python plot_supp_threshold_leak.py
    # reads results/recover_threshold_leak/rep*_*.csv if present (cluster
    # output, rsynced back) and caches them to notes/data/threshold_leak.tsv;
    # otherwise reads that committed TSV. Also reads notes/data/overview.json
    # for the real-data PCM contrast.
    # writes notes/figures/supp_threshold_leak.{pdf,png}
"""
import glob
import json
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figstyle

figstyle.apply()
HERE = op.dirname(op.abspath(__file__))
TSV = op.join(HERE, "notes/data/threshold_leak.tsv")
PERC_COLOR = "#2E7D6B"      # perceptual noise (teal, = RDM color elsewhere)
MEM_COLOR = "#B07C2B"       # memory noise
PCM_COLOR = "#C44E52"       # the real-data PCM value / "paradox" red

DOSE = ["a_x0", "a_x0.5", "full", "a_x1.5"]          # threshold dose-response
DOSE_X = [0.0, 0.5, 1.0, 1.5]
CONDS = ["full", "a_x0", "w0_x0", "noise_x0", "all_x0"]
COND_LABEL = {
    "full": "As fitted\n(everything)",
    "a_x0": "Threshold\ncontrast at 0",
    "w0_x0": "Urgency $w_0$\ncontrast at 0",
    "noise_x0": "Noise\ncontrasts at 0",
    "all_x0": "All contrasts\nat 0",
}


def load():
    files = sorted(glob.glob(op.join(HERE, "results/recover_threshold_leak/rep*_*.csv")))
    if files:
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        df.to_csv(TSV, sep="\t", index=False)
    else:
        df = pd.read_csv(TSV, sep="\t")
    return df


# 94%, not 95%: the same interval mass arviz's hdi() uses for the posterior
# HDIs in the other supplementary figures, so a reader never switches
# conventions mid-supplement.
HDI_PROB = 0.94


def ci(v, n_boot=20000, seed=0):
    """Bootstrap 94% CI of the mean (reps are exchangeable draws)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    bs = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    lo, hi = 100 * (1 - HDI_PROB) / 2, 100 * (1 + HDI_PROB) / 2
    return v.mean(), np.percentile(bs, lo), np.percentile(bs, hi)


def main():
    df = load()
    n_raw = len(df)
    df = df[df["max_rhat"] <= 1.05]
    # keep only reps that converged in EVERY condition, so all panels are paired
    ok = df.groupby("rep")["cond"].nunique()
    reps = ok[ok == df["cond"].nunique()].index
    df = df[df["rep"].isin(reps)]
    n_rep = df["rep"].nunique()
    S = json.load(open(op.join(HERE, "notes/data/overview.json")))
    real_pcm = S["choice"]["contrasts"]["perceptual_noise_sd"]["mean"]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.05, 1.15, 0.8]})
    axA, axB, axC = axes

    # ---------- Panel A: threshold dose-response ----------
    wide = df.pivot_table(index="rep", columns="cond",
                          values="rec_perceptual_noise_sd")[DOSE]
    true_perc = df[df["cond"] == "full"]["true_perceptual_noise_sd"].mean()
    for _, r in wide.iterrows():
        axA.plot(DOSE_X, r.values, color="0.75", lw=0.6, alpha=0.6, zorder=1)
    m = [ci(wide[c].dropna()) for c in DOSE]
    axA.errorbar(DOSE_X, [x[0] for x in m],
                 yerr=[[x[0] - x[1] for x in m], [x[2] - x[0] for x in m]],
                 color=PERC_COLOR, lw=2, marker="o", ms=6, capsize=3, zorder=4,
                 label="Recovered by the PCM")
    axA.axhline(true_perc, color="0.35", ls="-", lw=1.2, zorder=2)
    axA.annotate("True contrast (same in every condition)", (1.5, true_perc),
                 xytext=(0, 3), textcoords="offset points", fontsize=7.3,
                 color="0.3", ha="right", va="bottom")
    axA.axhline(0, ls=":", color="0.5", lw=0.9, zorder=1)
    axA.axhline(real_pcm, ls="--", color=PCM_COLOR, lw=1.2, zorder=2)
    axA.annotate("Real-data PCM", (0.02, real_pcm),
                 xycoords=("axes fraction", "data"), xytext=(0, 3),
                 textcoords="offset points", fontsize=7.3, color=PCM_COLOR,
                 ha="left", va="bottom")
    slopes = [np.polyfit(df[df["rep"] == r].set_index("cond").loc[DOSE, "true_a"].values,
                         wide.loc[r].values, 1)[0] for r in wide.index]
    axA.annotate(f"Slope = {np.mean(slopes):.2f} per unit of\nsimulated threshold contrast".replace("-", "\u2212"),
                 (0.03, 0.06), xycoords="axes fraction", fontsize=7.6, color="0.3",
                 ha="left", va="bottom")
    axA.set_xticks(DOSE_X)
    nat = df[df["cond"].isin(DOSE)].groupby("cond")[["nat_a_control", "nat_a_dyscalc"]].mean()
    axA.set_xticklabels([f"{x:g}×\n({nat.loc[c, 'nat_a_dyscalc']:.2f})" for x, c in zip(DOSE_X, DOSE)])
    axA.set_xlabel("Simulated threshold contrast\n(× fitted; mean dyscalc $a$ in brackets, "
                   f"control $a$ = {nat.loc['full', 'nat_a_control']:.2f})")
    axA.set_ylabel("Perceptual-noise contrast\nΔ (dyscalc − control), link scale")
    axA.margins(x=0.12)

    # ---------- Panel B: which contrast leaks? ----------
    y = np.arange(len(CONDS))[::-1]
    for i, c in enumerate(CONDS):
        sub = df[df["cond"] == c]
        for k, (par, col, off) in enumerate((("perceptual_noise_sd", PERC_COLOR, 0.16),
                                             ("memory_noise_sd", MEM_COLOR, -0.16))):
            mean, lo, hi = ci(sub[f"rec_{par}"])
            axB.plot([lo, hi], [y[i] + off] * 2, color=col, lw=2, solid_capstyle="butt",
                     zorder=3)
            axB.plot([mean], [y[i] + off], "o", color=col, ms=6, zorder=4,
                     label=("Perceptual" if k == 0 else "Memory") if i == 0 else None)
            axB.plot([sub[f"true_{par}"].mean()], [y[i] + off], marker="|", ms=9,
                     color=col, mew=1.6, zorder=5,
                     label=("True value" if (i == 0 and k == 0) else None))
    axB.axvline(0, ls=":", color="0.5", lw=0.9)
    axB.axvline(real_pcm, ls="--", color=PCM_COLOR, lw=1.2)
    axB.set_yticks(y)
    axB.set_yticklabels([COND_LABEL[c] for c in CONDS], fontsize=8)
    axB.set_xlabel("Noise contrast recovered by the PCM\nΔ (dyscalc − control), link scale")
    axB.set_ylim(y.min() - 0.6, y.max() + 0.6)
    axB.legend(loc="lower right", fontsize=7.5, ncol=1, handletextpad=0.4)

    # ---------- Panel C: what the PCM actually sees ----------
    for i, c in enumerate(["full", "a_x0", "noise_x0"]):
        sub = df[df["cond"] == c]
        for g in ("control", "dyscalc"):
            mean, lo, hi = ci(sub[f"sim_acc_{g}"])
            axC.errorbar([i], [mean], yerr=[[mean - lo], [hi - mean]],
                         color=figstyle.GROUP[g], marker="o", ms=6, capsize=3,
                         label=g.capitalize() if i == 0 else None)
        axC.plot([i, i], [sub["sim_acc_control"].mean(), sub["sim_acc_dyscalc"].mean()],
                 color="0.6", lw=1, zorder=1)
    axC.set_xticks([0, 1, 2])
    axC.set_xticklabels(["As\nfitted", "Threshold\nΔ at 0", "Noise\nΔ at 0"], fontsize=8)
    axC.set_xlim(-0.5, 2.5)
    axC.set_ylabel("Simulated accuracy")
    axC.set_xlabel("Simulated world")
    axC.legend(loc="best", fontsize=7.5)

    for ax, letter in zip(axes, "abc"):
        figstyle.panel_letter(ax, letter)
    fig.suptitle("The threshold difference is what the choice-only model reads "
                 "as a perceptual-noise difference", fontsize=10.5, y=1.06)
    import seaborn as sns
    sns.despine(fig=fig, offset=5, trim=True)
    axB.set_yticks(y)                       # despine(trim=True) resets fixed labels
    axB.set_yticklabels([COND_LABEL[c] for c in CONDS], fontsize=8)
    axB.spines["left"].set_visible(False)
    axB.tick_params(axis="y", length=0)
    out = op.join(HERE, "notes/figures/supp_threshold_leak.pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=200)
    print(f"Saved {out}")
    print(f"{n_rep} fully-paired reps ({n_raw} runs on disk)")


if __name__ == "__main__":
    main()
