"""Multipage PDF: one page per model, all group-level parameters in REAL space.

The hierarchical `*_mu` nodes are stored on the untransformed (link) scale and
treatment-coded ([Intercept, group[T.dyscalc]]). Here we map them back to the
natural parameter space the modeller cares about:

    control = T(Intercept)
    dyscalc = T(Intercept + group[T.dyscalc])

with T the parameter's own link (softplus for SDs / a / t0 / drifts, with a's
0.3 floor in the DDM; identity for prior_mu and w_s). Each parameter gets its
own little panel (own scale), control (blue) vs dyscalc (red), posterior mean
with 50% (thick) / 94% (thin) HDI, plus P(dyscalc > control).

  ~/mambaforge/envs/bauer/bin/python make_model_params_pdf.py
"""
import os.path as op
import warnings

import arviz as az
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import figstyle

warnings.filterwarnings("ignore")
figstyle.apply()
HERE = op.dirname(op.abspath(__file__))
GROUP = figstyle.GROUP                              # control blue, dyscalc red

# nice parameter labels + plotting order
PLABEL = {
    "perceptual_noise_sd": "σ perceptual", "memory_noise_sd": "σ memory",
    "n1_evidence_sd": "σ n1", "n2_evidence_sd": "σ n2", "evidence_sd": "σ evidence",
    "a": "Threshold a", "t0": "Non-decision t₀",
    "w_0": "Drift w₀", "w_d": "Drift w_d", "w_s": "Drift w_s",
    "prior_mu": "Prior μ", "prior_sd": "Prior σ",
}
ORDER = ["perceptual_noise_sd", "memory_noise_sd", "n1_evidence_sd", "n2_evidence_sd",
         "evidence_sd", "a", "t0", "w_0", "w_d", "w_s", "prior_mu", "prior_sd"]

MODELS = [
    ("Choice null (raw)",   ["model-choice_null_long_trace.nc"]),
    ("Choice σperc (raw)",  ["model-choice_ponly_long_trace.nc"]),
    ("Choice σmem (raw)",   ["model-choice_monly_long_trace.nc"]),
    ("Choice noise (raw)",  ["model-choice_group_long_trace.nc"]),
    ("Choice +prior (raw)", ["model-choice_full_trace.nc"]),
    ("Choice n1/n2 (raw)",  ["model-choice_n1n2_long_trace.nc"]),
    ("Choice null (KLW)",   ["model-choice_null_klw_trace.nc"]),
    ("Choice σperc (KLW)",  ["model-choice_ponly_klw_trace.nc"]),
    ("Choice σmem (KLW)",   ["model-choice_monly_klw_trace.nc"]),
    ("Choice noise (KLW)",  ["model-choice_group_klw_trace.nc"]),
    ("Choice +prior (KLW)", ["model-choice_full_klw_trace.nc"]),
    ("Choice n1/n2 (KLW)",  ["model-choice_n1n2_klw_trace.nc"]),
    ("DDM null",            ["model-ddm_null_long_trace.nc"]),
    ("DDM σperc",           ["model-ddm_ponly_long_trace.nc"]),
    ("DDM σmem",            ["model-ddm_monly_long_trace.nc"]),
    ("DDM noise",           ["model-ddm_pm_long_trace.nc"]),
    ("DDM noise +a",        ["model-ddm_group_long_trace.nc"]),
    ("DDM noise +a+t0",     ["model-ddm_t0group_long_trace.nc"]),
    ("DDM all free",        ["model-ddm_allfree_long_trace.nc"]),
    ("DDM n1/n2",           ["model-ddm_n1n2_long_trace.nc"]),
    ("RDM null",            ["model-rdm_null_long_trace.nc"]),
    ("RDM σperc",           ["model-rdm_ponly_long_trace.nc"]),
    ("RDM σmem",            ["model-rdm_monly_long_trace.nc"]),
    ("RDM noise",           ["model-rdm_pm_long_trace.nc"]),
    ("RDM noise +a",        ["model-rdm_group_long_trace.nc"]),
    ("RDM all free",        ["model-rdm_allfree_long_trace.nc"]),
    ("RDM n1/n2",           ["model-rdm_n1n2_long_trace.nc"]),
    ("RDM-no_ws null",      ["model-rdm_nows_null_long_trace.nc"]),
    ("RDM-no_ws σperc",     ["model-rdm_nows_ponly_long_trace.nc"]),
    ("RDM-no_ws σmem",      ["model-rdm_nows_monly_long_trace.nc"]),
    ("RDM-no_ws noise",     ["model-rdm_nows_pm_long_trace.nc"]),
]


def transform(arch, param, lin):
    """Map link-scale samples to natural parameter space."""
    if param in ("prior_mu", "w_s"):
        return lin
    floor = 0.3 if (param == "a" and arch == "ddm") else 0.0
    return floor + np.logaddexp(0.0, lin)            # min + softplus, stable


def real_space(post, arch, param):
    """Return (ctrl_samples, dys_samples, has_group) in natural space."""
    betas = post[f"{param}_mu"]
    rd = next(d for d in betas.dims if d.endswith("_regressors"))
    levels = list(post.coords[rd].values)
    icpt = betas.sel({rd: "Intercept"}).values.ravel()
    ctrl = transform(arch, param, icpt)
    grp = [l for l in levels if l != "Intercept"]
    if grp:
        dys_lin = icpt + betas.sel({rd: grp[0]}).values.ravel()
        return ctrl, transform(arch, param, dys_lin), True
    return ctrl, ctrl, False


def main():
    import matplotlib.pyplot as plt
    out = op.join(HERE, "notes", "figures", "model_parameters_all.pdf")
    n_pages = 0
    with PdfPages(out) as pdf:
        for label, cands in MODELS:
            p = next((op.join(HERE, "results", b) for b in cands
                      if op.exists(op.join(HERE, "results", b))), None)
            if p is None:
                continue
            arch = "ddm" if "DDM" in label else ("rdm" if "RDM" in label else "choice")
            idata = az.from_netcdf(p)
            post = idata.posterior
            params = [q for q in ORDER if f"{q}_mu" in post.data_vars]
            if not params:
                continue
            rh = az.summary(idata, var_names=[f"{q}_mu" for q in params])["r_hat"].max()
            ndiv = int(idata.sample_stats["diverging"].sum()) \
                if "diverging" in idata.sample_stats else -1

            n = len(params)
            fig, axes = plt.subplots(n, 1, figsize=(5.4, 0.92 * n + 0.7),
                                     squeeze=False, constrained_layout=True)
            axes = axes[:, 0]
            for ax, param in zip(axes, params):
                ctrl, dys, has_grp = real_space(post, arch, param)
                series = [("control", ctrl, 1.0), ("dyscalc", dys, 0.0)]
                for gname, s, y in series:
                    col = GROUP[gname]
                    lo94, hi94 = az.hdi(s, hdi_prob=0.94)
                    lo50, hi50 = az.hdi(s, hdi_prob=0.50)
                    ax.plot([lo94, hi94], [y, y], color=col, lw=1.3, solid_capstyle="round")
                    ax.plot([lo50, hi50], [y, y], color=col, lw=4.0, alpha=0.9,
                            solid_capstyle="round")
                    ax.scatter([s.mean()], [y], s=26, facecolor="white",
                               edgecolor=col, lw=1.3, zorder=5)
                ax.set_ylim(-0.6, 1.6)
                ax.set_yticks([1.0, 0.0])
                ax.set_yticklabels(["ctrl", "dys"], fontsize=7)
                ax.set_ylabel(PLABEL.get(param, param), rotation=0, ha="right",
                              va="center", fontsize=8.5, labelpad=2)
                if has_grp:
                    pgt = (dys > ctrl).mean()
                    ax.annotate(f"P(dys>ctrl)={pgt:.0%}", (0.99, 0.92),
                                xycoords="axes fraction", ha="right", va="top",
                                fontsize=6.3, color="0.45")
                else:
                    ax.annotate("(no group term)", (0.99, 0.92),
                                xycoords="axes fraction", ha="right", va="top",
                                fontsize=6.3, color="0.6")
            axes[-1].set_xlabel("Parameter value (natural space)", fontsize=8)
            fig.suptitle(f"{label}    [{post.sizes['chain']}×{post.sizes['draw']} "
                         f"draws · max r̂={rh:.3f} · div={ndiv}]", fontsize=9.5)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n_pages += 1
            print(f"  page: {label:22s} <- {op.basename(p)}")
    print(f"Saved {out}  ({n_pages} pages)")


if __name__ == "__main__":
    main()
