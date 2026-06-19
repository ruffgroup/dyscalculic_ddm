"""Model-comparison report PDF:
  page 1 - convergence/diagnostics table (all available models),
  page 2 - PSIS-LOO comparison within the choice+RT family (DDM/RDM, the only
           LOO-comparable set) + the choice-only family separately,
  page 3 - cross-model parameter synthesis (dyscalc-control per parameter).

    ~/mambaforge/envs/bauer/bin/python make_comparison.py
"""
import os.path as op
import warnings

import arviz as az
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import figstyle

warnings.filterwarnings("ignore")

# tag, arch, pretty
REGISTRY = [
    ("choice_monly", "choice", "Choice σmem only"),
    ("choice_ponly", "choice", "Choice σperc only"),
    ("choice_group", "choice", "Choice (noise)"),
    ("choice_full", "choice", "Choice + prior"),
    ("ddm_monly", "ddm", "DDM σmem only"),
    ("ddm_ponly", "ddm", "DDM σperc only"),
    ("ddm_pm", "ddm", "DDM noise"),
    ("ddm_group", "ddm", "DDM +a"),
    ("ddm_pmt0", "ddm", "DDM +t0"),
    ("ddm_t0group", "ddm", "DDM +a+t0"),
    ("ddm_sv_group", "ddm", "DDM-sv +a"),
    ("rdm_monly", "rdm", "RDM σmem only"),
    ("rdm_ponly", "rdm", "RDM σperc only"),
    ("rdm_pm", "rdm", "RDM noise"),
    ("rdm_group", "rdm", "RDM +a"),
    ("rdm_pmt0", "rdm", "RDM +t0"),
    ("rdm_pmat0", "rdm", "RDM +a+t0"),
    ("rdm_nows_ponly", "rdm", "RDM-no_ws σperc"),
    ("rdm_nows_monly", "rdm", "RDM-no_ws σmem"),
    # n1/n2-separate parameterization (raw per-stimulus encoding noise)
    ("choice_n1n2", "choice", "Choice n1/n2 sep"),
    ("ddm_n1n2", "ddm", "DDM n1/n2 sep"),
    ("rdm_n1n2", "rdm", "RDM n1/n2 sep"),
]
PARAMS = [("perceptual_noise_sd", "σ perceptual"), ("memory_noise_sd", "σ memory"),
          ("a", "Threshold a"), ("t0", "Non-decision t₀")]


def tpath(tag):
    for p in (f"results/model-{tag}_long_trace.nc", f"results/model-{tag}_trace.nc"):
        if op.exists(p):
            return p
    return None


def diag_row(idata):
    s = az.summary(idata)
    mu = s[s.index.str.contains("_mu")]
    nd = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    return mu.r_hat.max(), mu.ess_bulk.min(), nd


def get_coef(post, param):
    var = f"{param}_mu"
    if var not in post:
        return None
    rd = next((d for d in post[var].dims if d.endswith("_regressors")), None)
    if rd is None:
        return None
    gl = [l for l in post.coords[rd].values if l != "Intercept"]
    if not gl:
        return None
    da = post[var].sel({rd: gl[0]})
    dr = da.values.ravel()
    lo, hi = az.hdi(dr, hdi_prob=0.94)
    ess = float(az.ess(da.transpose("chain", "draw").values))
    return dr.mean(), lo, hi, ess


def main():
    figstyle.apply()
    import matplotlib.pyplot as plt
    avail = [(t, a, ti) for (t, a, ti) in REGISTRY if tpath(t)]
    idatas = {t: az.from_netcdf(tpath(t)) for t, _, _ in avail}
    print("available:", [t for t, *_ in avail])

    with PdfPages("notes/figures/report_comparison.pdf") as pdf:
        # ---- page 1: diagnostics table ----
        fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(avail) + 1.2))
        ax.axis("off")
        rows = []
        for t, a, ti in avail:
            rh, ess, nd = diag_row(idatas[t])
            conv = "ok" if (rh <= 1.01 and ess >= 400) else "SOFT"
            rows.append([ti, f"{rh:.3f}", f"{ess:.0f}", str(nd), conv])
        tab = ax.table(cellText=rows,
                       colLabels=["Model", "max r̂", "min ESS", "div", "conv"],
                       cellLoc="center", loc="center")
        tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.3)
        for (r, c), cell in tab.get_celld().items():
            cell.set_edgecolor("0.8")
            if r == 0:
                cell.set_text_props(fontweight="bold")
            elif rows[r - 1][4] == "SOFT":
                cell.set_facecolor("#F7E1E1")
        ax.set_title("Convergence diagnostics (group-level)", fontsize=11, pad=14)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ---- page 2: LOO within families (converged models only) ----
        def converged(idata):
            rh, ess, _ = diag_row(idata)
            return rh <= 1.05 and ess >= 400
        for fam, label in [(["ddm", "rdm"], "Choice + RT (DDM / RDM)"),
                           (["choice"], "Choice-only")]:
            grp = {ti: idatas[t] for t, a, ti in avail
                   if a in fam and "log_likelihood" in idatas[t].groups()
                   and converged(idatas[t])}
            if len(grp) < 2:
                continue
            cmp = az.compare(grp)
            fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(grp) + 1.6))
            az.plot_compare(cmp, ax=ax, insample_dev=False)
            ax.set_title(f"PSIS-LOO: {label}  (higher ELPD = better)", fontsize=10)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ---- page 3: parameter synthesis ----
        fig, axes = plt.subplots(1, len(PARAMS), figsize=(11, 3.2), sharey=True)
        ylabels = [ti for _, _, ti in avail]
        for ax, (param, plab) in zip(axes, PARAMS):
            for yi, (t, a, ti) in enumerate(avail):
                r = get_coef(idatas[t].posterior, param)
                if r is None:
                    continue
                mean, lo, hi, ess = r
                conv = ess >= 100
                col = "0.15" if conv else "0.6"
                ax.plot([lo, hi], [yi, yi], color=col, lw=2.0, solid_capstyle="round")
                ax.scatter([mean], [yi], s=28,
                           facecolor=(col if conv else "white"), edgecolor=col, lw=1.0)
                if not conv:
                    ax.annotate("*", (mean, yi), (0, 3), textcoords="offset points",
                                ha="center", fontsize=10, color="0.5")
            ax.axvline(0, ls=":", c="0.6", lw=0.8)
            ax.set_xlabel(f"{plab}\nΔ(dys−ctrl)", fontsize=8.5)
            ax.set_yticks(range(len(avail))); ax.set_yticklabels(ylabels, fontsize=7)
            ax.set_ylim(-0.6, len(avail) - 0.4)
        axes[0].annotate("* ESS<100", (0.03, 0.02), xycoords="axes fraction",
                         fontsize=6.5, color="0.5")
        fig.suptitle("Group effect across models (dyscalc − control)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig); plt.close(fig)
    print("Saved notes/figures/report_comparison.pdf")


if __name__ == "__main__":
    main()
