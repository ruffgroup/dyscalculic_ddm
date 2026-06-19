"""Why the group-SD prior matters: the HalfCauchy tail is running away.

(a) prior densities: HalfCauchy(0.25) vs HalfNormal(1.0) on the group SD —
    the Cauchy puts heavy mass far out; HalfNormal does not.
(b) realized group-SD posteriors (link scale) for the converged ddm_group vs
    the broken ddm_pm — the broken fit's memory/perceptual group SDs balloon
    into the Cauchy tail (the funnel that traps chains; ESS≈7).

  ~/mambaforge/envs/bauer/bin/python make_group_sd_diag.py
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
SD_PARS = ["perceptual_noise_sd_sd", "memory_noise_sd_sd", "a_sd", "t0_sd"]
PLAB = {"perceptual_noise_sd_sd": "σ-σ perceptual", "memory_noise_sd_sd": "σ-σ memory",
        "a_sd": "σ-σ threshold a", "t0_sd": "σ-σ t₀"}


def sd_intercept(idata, par):
    """Posterior samples of the group-SD (intercept regressor if present)."""
    if par not in idata.posterior:
        return None
    da = idata.posterior[par]
    rd = next((d for d in da.dims if d.endswith("_regressors")), None)
    if rd is not None:
        da = da.sel({rd: "Intercept"})
    return da.values.ravel()


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    # ---- (a) prior tails ----
    ax = axes[0]
    x = np.linspace(0, 12, 800)
    halfcauchy = (2 / (np.pi * 0.25)) / (1 + (x / 0.25) ** 2)
    halfnormal = np.sqrt(2 / (np.pi * 1.0 ** 2)) * np.exp(-x ** 2 / (2 * 1.0 ** 2))
    ax.plot(x, halfcauchy, color="#C44E52", lw=1.6, label="HalfCauchy(0.25)")
    ax.plot(x, halfnormal, color="#3B5BA5", lw=1.6, label="HalfNormal(1.0)")
    ax.fill_between(x, halfcauchy, where=(x > 2), color="#C44E52", alpha=0.18, lw=0)
    ax.set_xlim(0, 12); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Group SD (link scale)"); ax.set_ylabel("Prior density")
    ax.set_title("Prior on the group SD", fontsize=9.5)
    ax.annotate("Cauchy keeps\nheavy mass out here", (5.0, 0.06), fontsize=7,
                color="#C44E52", va="bottom")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    figstyle.panel_letter(ax, "a")

    # ---- (b) realized posteriors: converged vs broken ----
    ax = axes[1]
    fits = [("ddm_group (converged)", "results/model-ddm_group_long_trace.nc", "#3B5BA5"),
            ("ddm_pm (broken, ESS≈7)", "results/model-ddm_pm_long_trace.nc", "#C44E52")]
    ylab = []
    for yi, par in enumerate(SD_PARS):
        ylab.append(PLAB[par])
        for j, (lab, path, col) in enumerate(fits):
            s = sd_intercept(az.from_netcdf(op.join(HERE, path)), par)
            if s is None:
                continue
            y = yi + (0.18 if j == 0 else -0.18)
            lo, hi = np.percentile(s, [2.5, 97.5])
            ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="round")
            ax.scatter([np.median(s)], [y], s=22, facecolor="white", edgecolor=col,
                       lw=1.3, zorder=5)
            ax.annotate(f"{hi:.1f}", (hi, y), (4, 0), textcoords="offset points",
                        va="center", fontsize=6, color=col)
    ax.set_yticks(range(len(SD_PARS))); ax.set_yticklabels(ylab, fontsize=8)
    ax.set_xlabel("Group-SD posterior (median, 95%)"); ax.set_xlim(left=0)
    ax.set_title("Realized group SDs (link scale)", fontsize=9.5)
    ax.annotate("ddm_group", (0.97, 0.97), xycoords="axes fraction", ha="right",
                va="top", color="#3B5BA5", fontsize=7.5, fontweight="bold")
    ax.annotate("ddm_pm (broken)", (0.97, 0.89), xycoords="axes fraction", ha="right",
                va="top", color="#C44E52", fontsize=7.5, fontweight="bold")
    figstyle.panel_letter(ax, "b")

    fig.suptitle("The HalfCauchy group-SD tail runs away in the broken DDM fits",
                 fontsize=10)
    figstyle.finish(fig, op.join(HERE, "notes", "figures", "group_sd_diagnostic.pdf"))


if __name__ == "__main__":
    main()
