"""Clean key-models figure: where does the dyscalculia deficit load?

Only the three KEY model classes, each from its long, converged chain:
  Choice (KLW)  - static cumulative-normal choice, KLW-consistent noise
  DDM           - drift-diffusion (Wiener)
  RDM           - race / Wald accumulator

For each we plot the group contrast Δ(dyscalc - control) on the perceptual
and memory encoding-noise SDs (+ the decision threshold a, accumulators only),
as posterior mean with 50% / 94% HDI.

NB cross-model magnitude is NOT directly comparable (each link has its own
scale: choice cumulative-normal vs DDM drift vs RDM rate). The message is the
SIGN/direction within each model -- the perceptual-vs-memory dissociation.

  ~/mambaforge/envs/bauer/bin/python make_key_figures.py
"""
import os.path as op
import warnings

import arviz as az
import numpy as np
import matplotlib.pyplot as plt

import figstyle

warnings.filterwarnings("ignore")
figstyle.apply()
HERE = op.dirname(op.abspath(__file__))

# key model -> converged long trace + display colour. Two matched comparisons:
#   choice raw vs KLW (decision-noise normalization); RDM +w_s vs -w_s.
KEY = [
    ("Choice (raw)",  "results/model-choice_group_long_trace.nc",    "#B0B0B0"),
    ("Choice (KLW)",  "results/model-choice_group_klw_trace.nc",     "#555555"),
    ("DDM",           "results/model-ddm_group_long_trace.nc",       "#3B5BA5"),
    ("RDM +$w_s$",  "results/model-rdm_pm_long_trace.nc",       "#C44E52"),
    ("RDM −$w_s$",  "results/model-rdm_nows_pm_long_trace.nc",  "#E0936F"),
]
PARAMS = [("perceptual_noise_sd", "σ perceptual"),
          ("memory_noise_sd", "σ memory")]


def contrast_samples(post, param):
    """Flat posterior samples of the group Δ (first non-Intercept regressor)."""
    var = f"{param}_mu"
    if var not in post:
        return None
    rd = next((d for d in post[var].dims if d.endswith("_regressors")), None)
    if rd is None:
        return None
    levels = [l for l in post.coords[rd].values if l != "Intercept"]
    if not levels:
        return None
    return post[var].sel({rd: levels[0]}).values.ravel()


def main():
    idatas = {name: az.from_netcdf(op.join(HERE, p)) for name, p, _ in KEY}
    colors = {name: c for name, _, c in KEY}
    names = [n for n, _, _ in KEY]

    fig, axes = plt.subplots(1, len(PARAMS), figsize=(5.8, 2.8), sharey=True,
                             constrained_layout=True)
    for ax, (param, plab) in zip(axes, PARAMS):
        for yi, name in enumerate(names):
            s = contrast_samples(idatas[name].posterior, param)
            if s is None:                       # e.g. choice has no threshold
                ax.scatter([], [])
                continue
            mean = s.mean()
            lo94, hi94 = az.hdi(s, hdi_prob=0.94)
            lo50, hi50 = az.hdi(s, hdi_prob=0.50)
            col = colors[name]
            pgt0 = (s > 0).mean()
            ax.plot([lo94, hi94], [yi, yi], color=col, lw=1.4, solid_capstyle="round")
            ax.plot([lo50, hi50], [yi, yi], color=col, lw=4.0, solid_capstyle="round",
                    alpha=0.9)
            ax.scatter([mean], [yi], s=30, facecolor="white", edgecolor=col,
                       lw=1.4, zorder=5)
            # direction tag (P>0) just past the interval
            ax.annotate(f"{pgt0:.0%}", (hi94, yi), (4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=6.5, color=col)
        # faint bands group the two matched comparisons (choice pair; RDM pair)
        ax.axhspan(-0.5, 1.5, color="0.93", zorder=0)     # Choice raw + KLW
        ax.axhspan(2.5, 4.5, color="0.93", zorder=0)      # RDM +wₛ / −wₛ
        ax.axvline(0, ls=":", c="0.55", lw=0.9)
        ax.set_xlabel(f"{plab}\nΔ (dyscalc − control)", fontsize=9)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.margins(x=0.22)
    fig.suptitle("Where does the deficit load?  Group Δ per key model "
                 "(long converged chains)", fontsize=10)
    fig.text(0.5, -0.04, "Posterior mean · 50% (thick) / 94% (thin) HDI · "
             "% = P(Δ>0) · magnitudes not comparable across model classes",
             ha="center", fontsize=6.8, color="0.4")
    figstyle.finish(fig, op.join(HERE, "notes", "figures", "key_models_parameters.pdf"))


if __name__ == "__main__":
    main()
