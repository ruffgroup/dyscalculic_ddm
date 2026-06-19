"""Mechanistic sanity check: how does the compression (stake/size) effect
depend on perceptual vs memory noise?

Uses the Bayesian-observer choice rule shared by ALL the models (static, DDM,
RDM differ only in the response layer; the front-end is identical). Each noisy
log-magnitude estimate is shrunk toward the prior by its reliability weight;
the comparison of the two shrunken estimates gives P(chose n2).

    m_hat_i = w_i x_i + (1-w_i) mu_p,   w_i = sp^2 / (sp^2 + sigma_i^2)
    sigma(n2) = s_perc                  (just-seen)
    sigma(n1) = s_perc + s_mem          (remembered; bauer's linear sum)
    DV = m_hat2 - m_hat1
    P(chose n2) = Phi( E[DV] / sd[DV] )

Compression slope (the size effect) ~ (w2 - w1): raising s_perc pushes
sigma(n1)/sigma(n2) -> 1 (more symmetric) -> SMALLER compression; raising s_mem
-> LARGER compression. We overlay the ddm_group-fitted control vs dyscalc
points to show the fitted move (perceptual up) goes the WRONG way for the
observed larger dyscalc compression.

    ~/mambaforge/envs/bauer/bin/python sim_compression.py \
        --out notes/figures/sim_compression.png
"""
import argparse
import os
import os.path as op

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from bauer.utils.bayes import softplus

PAL = {"control": "#4C72B0", "dyscalc": "#C44E52"}


def reliability(sp_prior2, sigma):
    return sp_prior2 / (sp_prior2 + sigma ** 2)


def p_chose_n2(m1, m2, s_perc, s_mem, mu_p, sp_prior):
    s1, s2 = s_perc + s_mem, s_perc
    sp2 = sp_prior ** 2
    w1, w2 = reliability(sp2, s1), reliability(sp2, s2)
    e_dv = w2 * m2 - w1 * m1 + (w1 - w2) * mu_p
    sd_dv = np.sqrt(w2 ** 2 * s2 ** 2 + w1 ** 2 * s1 ** 2)
    return stats.norm.cdf(e_dv / sd_dv)


def compression_slope(design, s_perc, s_mem, mu_p, sp_prior):
    """Realized slope of predicted P(chose n2) on log(n1) over the real design."""
    m1 = np.log(design["n1"].values)
    m2 = np.log(design["n2"].values)
    p = p_chose_n2(m1, m2, s_perc, s_mem, mu_p, sp_prior)
    # weight by trial counts; slope of p on log(n1)
    sl = np.polyfit(m1, p, 1, w=design["w"].values)[0]
    return sl


def main(out):
    df = pd.read_csv("data/magjudge_behavior_DNumRisk.csv")
    df = df[df["rt"] >= 0.20]
    design = df.groupby(["n1", "n2"]).size().rename("w").reset_index()
    logs = np.log(np.concatenate([df["n1"], df["n2"]]))
    mu_p, sp_prior = logs.mean(), logs.std()
    print(f"empirical prior: mu={mu_p:.3f}, sd={sp_prior:.3f}")

    # ddm_group fitted group means (pre-softplus), from analyze_fit.py:
    fit = {"control": dict(sp=softplus(-1.620), sm=softplus(-2.405)),
           "dyscalc": dict(sp=softplus(-1.620 + 0.252), sm=softplus(-2.405 + 0.057))}
    for g, v in fit.items():
        print(f"{g}: s_perc={v['sp']:.3f}  s_mem={v['sm']:.3f}  "
              f"ratio s1/s2={1 + v['sm']/v['sp']:.3f}")

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: compression slope vs perceptual noise, for several memory noises.
    sp_grid = np.linspace(0.05, 0.6, 60)
    for sm in [0.05, 0.10, 0.20, 0.35]:
        sl = [compression_slope(design, sp, sm, mu_p, sp_prior) for sp in sp_grid]
        axes[0].plot(sp_grid, sl, label=f"σ_mem = {sm:.2f}")
    for g, v in fit.items():
        sl = compression_slope(design, v["sp"], v["sm"], mu_p, sp_prior)
        axes[0].scatter([v["sp"]], [sl], color=PAL[g], s=110, zorder=5,
                        edgecolor="k", label=f"{g} (fitted)")
    axes[0].set_xlabel("σ perceptual"); axes[0].set_ylabel("Compression slope\n(dP(chose n2)/dlog n1)")
    axes[0].set_title("σ_memory is a STRONG lever; σ_perceptual is\nweak & non-monotonic (fitted pts ≈ flat peak)")
    axes[0].legend(fontsize=8)

    # Panel 2: predicted P(chose n2) vs n1 (fitted params) overlaid on DATA.
    obs = df.copy()
    if obs["group"].dtype != object:
        obs["group"] = obs["group"].map({0: "control", 1: "dyscalc"})
    obs_g = obs.groupby(["group", "n1"])["chose_n2"].mean()
    n1s = np.sort(design["n1"].unique())
    for g, v in fit.items():
        ys = []
        for n1 in n1s:
            d = design[design.n1 == n1]
            p = p_chose_n2(np.log(d.n1.values), np.log(d.n2.values),
                           v["sp"], v["sm"], mu_p, sp_prior)
            ys.append(np.average(p, weights=d.w.values))
        axes[1].plot(n1s, ys, "-", color=PAL[g], lw=2, label=f"{g} (model)")
        axes[1].scatter(obs_g[g].index, obs_g[g].values, color=PAL[g], s=55,
                        edgecolor="k", zorder=5, label=f"{g} (data)")
    axes[1].axhline(0.5, ls=":", c="grey")
    axes[1].set_xlabel("n1 (reference)"); axes[1].set_ylabel("P(chose n2)")
    axes[1].set_title("Model captures only ~1/3 of the data's\ngroup gap in compression (underfits)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Compression is driven efficiently by σ_memory / prior — not σ_perceptual",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(op.dirname(op.abspath(out)), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notes/figures/sim_compression.png")
    args = p.parse_args()
    main(args.out)
