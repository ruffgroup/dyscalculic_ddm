"""Full group-level parameter figure for the RDM (or any model): one panel per
free parameter, on its natural scale (transform read from the model), control
vs dyscalc where the parameter is regressed on group, single posterior
otherwise. House style.

    ~/mambaforge/envs/bauer/bin/python plot_rdm_params.py \
        results/model-rdm_group_trace.nc --model rdm_group \
        --out notes/figures/rdm_params.pdf
"""
import argparse

import arviz as az
import numpy as np
import pandas as pd

import figstyle
from fit_ddm import build_model

LABELS = {"perceptual_noise_sd": "σ perceptual", "memory_noise_sd": "σ memory",
          "a": "Threshold a", "t0": "Non-decision t₀",
          "w_0": "Drift w0 (offset)", "w_d": "Drift wd (difference)",
          "w_s": "Drift ws (sum)"}


def transform_fn(name):
    return {"softplus": lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0),
            "logistic": lambda x: 1 / (1 + np.exp(-x)),
            "identity": lambda x: x}[name]


def load_df():
    df = pd.read_csv("data/magjudge_behavior_DNumRisk.csv")
    df = df[df["rt"] >= 0.20].copy()
    df["choice"] = df["choice"].astype(bool)
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    return df.set_index("subject")


def main(nc, model_label, out):
    figstyle.apply()
    import matplotlib.pyplot as plt
    model = build_model(model_label, load_df())
    fp = model.free_parameters
    post = az.from_netcdf(nc).posterior

    names = [n for n in fp if f"{n}_mu" in post]
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        tf = transform_fn(fp[name].get("transform", "identity"))
        var = f"{name}_mu"
        rdim = next(d for d in post[var].dims if d.endswith("_regressors"))
        levels = list(post.coords[rdim].values)
        icpt = post[var].sel({rdim: "Intercept"}).values.ravel()
        gl = [l for l in levels if l != "Intercept"]
        series = [("control", tf(icpt), figstyle.GROUP["control"])]
        if gl:
            series.append(("dyscalc", tf(icpt + post[var].sel({rdim: gl[0]}).values.ravel()),
                           figstyle.GROUP["dyscalc"]))
        else:
            series = [("", tf(icpt), figstyle.NEUTRAL)]
        for x, (lab, draws, col) in enumerate(series):
            lo, hi = az.hdi(draws, hdi_prob=0.94)
            ax.plot([x, x], [lo, hi], color=col, lw=2.5, solid_capstyle="round")
            ax.scatter([x], [draws.mean()], color=col, s=26, zorder=3)
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([s[0].capitalize() for s in series], fontsize=7)
        ax.set_xlim(-0.6, len(series) - 0.4)
        ax.set_xlabel(LABELS.get(name, name), fontsize=8.5)
        if gl:
            p = float((series[1][1] > series[0][1]).mean())
            ax.annotate(f"P(D>C)={p:.2f}", xy=(0.5, 0.99), xycoords="axes fraction",
                        ha="center", va="top", fontsize=6.5, color="0.3")
    for ax in axes[len(names):]:
        ax.set_visible(False)
    fig.supylabel("Posterior (natural units)", fontsize=9)
    figstyle.finish(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("nc")
    p.add_argument("--model", default="rdm_group")
    p.add_argument("--out", default="notes/figures/rdm_params.pdf")
    args = p.parse_args()
    main(args.nc, args.model, args.out)
