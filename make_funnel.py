"""Show the funnel.

Neal's funnel for the memory-noise hierarchy: y = log(group SD), x = a subject's
random deviation (= group_sd · offset). As the group SD shrinks, the subject
deviations are squeezed toward 0 (the neck); as it grows, they fan out (the
mouth). NUTS can't pick one step size for both -> chains stick. Points colored
by chain: in the broken ddm_pm the chains occupy disjoint bands (stuck, ESS≈7);
in the converged ddm_group they overlap (mixed).

  ~/mambaforge/envs/bauer/bin/python make_funnel.py
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
PAR = "memory_noise_sd"


def pieces(idata, par, n_subj=12, thin=4):
    """Pool several subjects' deviations to trace the funnel envelope."""
    post = idata.posterior

    def icpt(da):
        rd = next((d for d in da.dims if d.endswith("_regressors")), None)
        return da.sel({rd: "Intercept"}) if rd else da

    sd_i = icpt(post[f"{par}_sd"])                    # (chain, draw)
    off = icpt(post[f"{par}_offset"])                 # (chain, draw, subject)
    nchain, ndraw = sd_i.shape
    log_sd = np.log(sd_i.values)[:, ::thin]
    nsub = min(n_subj, off.sizes["subject"])
    devs, logs, cids = [], [], []
    for s in range(nsub):
        dev = (sd_i.values * off.isel(subject=s).values)[:, ::thin]
        devs.append(dev.ravel()); logs.append(np.repeat(log_sd, 1).ravel() if False else log_sd.ravel())
        cids.append(np.repeat(np.arange(nchain), log_sd.shape[1]))
    return np.concatenate(devs), np.concatenate(logs), np.concatenate(cids)


def main():
    fits = [("ddm_group (converged, ESS≈12k)", "results/model-ddm_group_long_trace.nc"),
            ("ddm_pm (broken, ESS≈7)", "results/model-ddm_pm_long_trace.nc")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True,
                             constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for ax, (lab, path) in zip(axes, fits):
        dev, log_sd, cid = pieces(az.from_netcdf(op.join(HERE, path)), PAR)
        for c in np.unique(cid):
            m = cid == c
            ax.scatter(dev[m], log_sd[m], s=2, color=cmap(c), alpha=0.25, lw=0,
                       rasterized=True)
        ax.set_xlabel("Subject random deviation  (group σ · offset)")
        ax.set_title(lab, fontsize=9)
    axes[0].set_ylabel("log( group SD )  [memory noise]")
    axes[1].annotate("chains in disjoint\nbands = stuck", (0.5, 0.97),
                     xycoords="axes fraction", ha="center", va="top",
                     fontsize=7.5, color="0.25")
    axes[0].annotate("chains overlap\n= mixed", (0.5, 0.03),
                     xycoords="axes fraction", ha="center", va="bottom",
                     fontsize=7.5, color="0.25")
    fig.suptitle("Neal's funnel: memory-noise group SD vs subject deviation "
                 "(color = MCMC chain)", fontsize=9.5)
    figstyle.finish(fig, op.join(HERE, "notes", "figures", "funnel.pdf"), trim=False)


if __name__ == "__main__":
    main()
