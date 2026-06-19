"""Group-level parameter posteriors (control vs dyscalc) for any fitted model.

Auto-discovers the group-level regression parameters (vars ending in `_mu`
with a regressor dim), transforms each to its natural scale, and shows the
per-group posterior (point = mean, line = 94% HDI). House style throughout.

    ~/mambaforge/envs/bauer/bin/python plot_group_contrast.py \
        results/model-choice_group_trace.nc --out notes/figures/choice_contrast.pdf
"""
import argparse
import os.path as op

import arviz as az
import numpy as np

import figstyle
from bauer.utils.bayes import softplus

# Pretty labels for the parameters we care about (others are auto-included).
LABELS = {"perceptual_noise_sd": "σ perceptual\n(number sense)",
          "memory_noise_sd": "σ memory", "a": "Threshold a",
          "t0": "Non-decision t₀", "prior_mu": "Prior mean μₚ",
          "prior_sd": "Prior width σₚ"}
# Positive-constrained params are fit on a softplus (pre-transform) scale.
def _transform(name, x):
    pos = (name.endswith("_sd") or "noise" in name or name in ("a", "t0")
           or name.startswith("w_"))
    return softplus(x) if pos else x


def per_group(post, var):
    rdim = next(d for d in post[var].dims if d.endswith("_regressors"))
    levels = list(post.coords[rdim].values)
    grp = next(l for l in levels if l != "Intercept")
    icpt = post[var].sel({rdim: "Intercept"}).values.ravel()
    base = var[:-3]  # strip _mu
    c = _transform(base, icpt)
    d = _transform(base, icpt + post[var].sel({rdim: grp}).values.ravel())
    return c, d


def main(nc, out):
    figstyle.apply()
    import matplotlib.pyplot as plt
    post = az.from_netcdf(nc).posterior
    reg_vars = [v for v in post.data_vars
                if v.endswith("_mu") and not v.endswith("_mu_mu")
                and any(d.endswith("_regressors") for d in post[v].dims)
                and len(post.coords[next(d for d in post[v].dims
                                         if d.endswith("_regressors"))]) > 1]
    # stable, meaningful order
    order = ["perceptual_noise_sd", "memory_noise_sd", "a", "t0",
             "prior_mu", "prior_sd"]
    reg_vars = sorted(reg_vars, key=lambda v: order.index(v[:-3])
                      if v[:-3] in order else 99)

    n = len(reg_vars)
    fig, axes = plt.subplots(1, n, figsize=(1.7 * n + 0.6, 3.0))
    if n == 1:
        axes = [axes]
    for i, (ax, v) in enumerate(zip(axes, reg_vars)):
        c, d = per_group(post, v)
        for x, (name, draws) in enumerate([("control", c), ("dyscalc", d)]):
            lo, hi = az.hdi(draws, hdi_prob=0.94)
            ax.plot([x, x], [lo, hi], color=figstyle.GROUP[name], lw=2.5,
                    solid_capstyle="round", zorder=2)
            ax.scatter([x], [draws.mean()], color=figstyle.GROUP[name], s=28,
                       zorder=3)
        p = float((d > c).mean())
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Dyscalc"] if i == 0 else ["", ""])
        ax.set_xlim(-0.6, 1.6)
        ax.set_xlabel(LABELS.get(v[:-3], v[:-3]), fontsize=9)
        # in-panel effect annotation
        ax.annotate(f"P(D>C)={p:.2f}", xy=(0.5, 0.98), xycoords="axes fraction",
                    ha="center", va="top", fontsize=7, color="0.3")
    axes[0].set_ylabel("Posterior (natural units)")
    figstyle.finish(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("nc")
    p.add_argument("--out", default="notes/figures/group_contrast.pdf")
    args = p.parse_args()
    main(args.nc, args.out)
