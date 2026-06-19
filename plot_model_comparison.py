"""Cross-model synthesis: the dyscalc - control group coefficient for each
cognitive parameter, across models. Shows how the attribution of the deficit
shifts with the response model (choice vs DDM vs DDM+t0 vs RDM). Coefficients
are on each parameter's linear-predictor (pre-transform) scale, so they're
directly comparable across models. Estimates with ESS < 100 (not converged)
are drawn faded + open, with an asterisk.

    ~/mambaforge/envs/bauer/bin/python plot_model_comparison.py \
        --out notes/figures/model_comparison.pdf
"""
import argparse
import os.path as op

import arviz as az
import numpy as np

import figstyle

MODELS = [("choice_monly", "Choice σmem-only"),
          ("choice_ponly", "Choice σperc-only"),
          ("choice_group", "Choice (noise)"),
          ("ddm_ponly", "DDM σperc-only"),
          ("ddm_pm", "DDM noise"),
          ("ddm_group", "DDM +a"),
          ("ddm_pmt0", "DDM +t0"),
          ("rdm_monly", "RDM σmem-only"),
          ("rdm_ponly", "RDM σperc-only"),
          ("rdm_pm", "RDM noise"),
          ("rdm_group", "RDM +a"),
          ("rdm_pmt0", "RDM +t0"),
          ("rdm_pmat0", "RDM +a+t0")]
PARAMS = [("perceptual_noise_sd", "σ perceptual"), ("memory_noise_sd", "σ memory"),
          ("a", "Threshold a"), ("t0", "Non-decision t₀")]


def get_coef(post, param):
    """(mean, lo, hi, ess) of the dyscalc group coefficient, or None."""
    var = f"{param}_mu"
    if var not in post:
        return None
    rdim = next((d for d in post[var].dims if d.endswith("_regressors")), None)
    if rdim is None:
        return None
    gl = [l for l in post.coords[rdim].values if l != "Intercept"]
    if not gl:
        return None
    da = post[var].sel({rdim: gl[0]})
    draws = da.values.ravel()
    lo, hi = az.hdi(draws, hdi_prob=0.94)
    ess = float(az.ess(da.transpose("chain", "draw").values))  # (chain,draw)->float
    return draws.mean(), lo, hi, ess


def main(out):
    figstyle.apply()
    import matplotlib.pyplot as plt
    traces = {}
    for m, _ in MODELS:
        f = next((p for p in (f"results/model-{m}_long_trace.nc",
                              f"results/model-{m}_trace.nc") if op.exists(p)), None)
        if f:
            traces[m] = az.from_netcdf(f).posterior

    fig, axes = plt.subplots(1, len(PARAMS), figsize=(11, 3.0), sharey=True)
    ylabels = [lab for _, lab in MODELS]
    for ax, (param, plab) in zip(axes, PARAMS):
        for yi, (m, mlab) in enumerate(MODELS):
            if m not in traces:
                continue
            r = get_coef(traces[m], param)
            if r is None:
                continue
            mean, lo, hi, ess = r
            conv = ess >= 100
            col = "0.15" if conv else "0.6"
            ax.plot([lo, hi], [yi, yi], color=col, lw=2.2,
                    solid_capstyle="round", zorder=2)
            ax.scatter([mean], [yi], s=34, facecolor=(col if conv else "white"),
                       edgecolor=col, lw=1.0, zorder=3)
            if not conv:
                ax.annotate("*", xy=(mean, yi), xytext=(0, 3),
                            textcoords="offset points", ha="center",
                            fontsize=11, color="0.5")
        ax.axvline(0, ls=":", c="0.6", lw=0.8, zorder=0)
        ax.set_xlabel(f"{plab}\nΔ (dyscalc − control)", fontsize=9)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(ylabels)
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
    axes[0].annotate("* ESS<100\n(not converged)", xy=(0.03, 0.03),
                     xycoords="axes fraction", fontsize=6.5, color="0.5",
                     va="bottom")
    figstyle.finish(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notes/figures/model_comparison.pdf")
    args = p.parse_args()
    main(args.out)
