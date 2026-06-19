"""Tiebreaker for the rate-vs-noise interpretation (LOO is 50/50).

The noise-deficit model (σ~group) and the rate-deficit model (w_d~group) fit
equally well, because both encode the same SNR deficit. But they make a
DIFFERENT prediction about COMPRESSION: σ enters the reliability weight
w = σ_p²/(σ_p²+σ²) (more noise -> more regression to the prior -> more
compression), whereas w_d is a flat gain that cannot touch compression. So:

  observed group compression difference reproduced by the NOISE model  -> noise
  observed group compression difference MISSED by the RATE model       -> not rate

We plot P(chose n2) vs n1 by group: data (points) vs each model's PPC (lines),
side by side. Whichever model reproduces the control-vs-dyscalc *gap* wins the
interpretation.

  ~/mambaforge/envs/bauer/bin/python make_rate_vs_noise_ppc.py
"""
import os.path as op
import warnings

import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle
import fit_ddm

warnings.filterwarnings("ignore")
figstyle.apply()
HERE = op.dirname(op.abspath(__file__))
GROUP = figstyle.GROUP
N_PPC = 60

# (panel title, build spec, trace)
KEY = [
    ("Noise-deficit (σ~group)", "rdm:perceptual_noise_sd,memory_noise_sd",
     "results/model-rdm_noisedef_hn_trace.nc"),
    ("Rate-deficit (w_d~group)", "rdm:w_d",
     "results/model-rdm_ratedef_hn_trace.nc"),
]


def load_df():
    df = pd.read_csv(op.join(HERE, "data", "magjudge_behavior_DNumRisk.csv"))
    df = df[df["rt"] >= 0.20].copy()
    df["choice"] = df["chose_n2"].astype(bool)
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    return df.set_index(["subject", "run", "trial_nr"])


def predicted(df, spec, trace):
    m = fit_ddm.build_model(spec, df)
    m.group_sd_dist = "halfnormal"
    m.build_estimation_model(df, hierarchical=True)
    idata = az.from_netcdf(op.join(HERE, trace))
    sim = m.ppc(df, idata, n_posterior_samples=N_PPC, progressbar=False)
    sim = sim.join(df[["n1", "group"]])
    per = (sim.groupby(["ppc_sample", "group", "n1"])["simulated_choice"]
              .mean().reset_index())
    rows = []
    for (g, n1), s in per.groupby(["group", "n1"]):
        v = s["simulated_choice"].values
        lo, hi = az.hdi(v, hdi_prob=0.94)
        rows.append(dict(group=g, n1=n1, med=np.median(v), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def main():
    df = load_df()
    obs = (df.reset_index().groupby(["group", "n1"])["choice"].mean()
             .rename("p").reset_index())

    fig, axes = plt.subplots(1, len(KEY), figsize=(7.2, 3.0), sharey=True,
                             constrained_layout=True)
    for ax, (title, spec, trace) in zip(axes, KEY):
        pred = predicted(df, spec, trace)
        for g, col in GROUP.items():
            pr = pred[pred.group == g].sort_values("n1")
            ob = obs[obs.group == g].sort_values("n1")
            ax.fill_between(pr.n1, pr.lo, pr.hi, color=col, alpha=0.22, lw=0)
            ax.plot(pr.n1, pr.med, color=col, lw=1.3, zorder=3)
            ax.scatter(ob.n1, ob.p, s=26, facecolor="white", edgecolor=col,
                       lw=1.3, zorder=4)
        ax.axhline(0.5, ls=":", c="0.6", lw=0.8)
        ax.set_xticks([5, 10, 15, 20]); ax.set_xlabel("n1 (first number)")
        ax.set_title(title, fontsize=9.5)
    axes[0].set_ylabel("P(chose n2)"); axes[0].set_ylim(0.25, 0.65)
    axes[0].annotate("Control", (0.05, 0.16), xycoords="axes fraction",
                     color=GROUP["control"], fontsize=8, fontweight="bold")
    axes[0].annotate("Dyscalc", (0.05, 0.05), xycoords="axes fraction",
                     color=GROUP["dyscalc"], fontsize=8, fontweight="bold")
    fig.suptitle("Rate vs noise tiebreaker: does each model reproduce the GROUP "
                 "compression gap?  (pts=data, line/band=model PPC)", fontsize=9)
    figstyle.finish(fig, op.join(HERE, "notes", "figures", "rate_vs_noise_ppc.pdf"))

    # quantify the group gap (dyscalc - control) in P(chose n2) at the extreme n1s
    print("Group gap [P(chose n2 | dyscalc) - control] by n1:")
    o = obs.pivot(index="n1", columns="group", values="p")
    o["gap_DATA"] = o["dyscalc"] - o["control"]
    for ax, (title, spec, trace) in zip(axes, KEY):
        pr = predicted(df, spec, trace).pivot(index="n1", columns="group", values="med")
        o[f"gap_{title.split()[0]}"] = pr["dyscalc"] - pr["control"]
    print(o.round(3))


if __name__ == "__main__":
    main()
