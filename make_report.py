"""Multi-page report PDFs across all fitted models:
  - params PDF: one page per model, every free parameter control vs dyscalc
    (natural scale, 94% HDI, P(D>C) + convergence flag).
  - ppc PDF: one page per model, P(chose n2) vs n1 (data vs model band); plus a
    Ratcliff QPP for the RT models (DDM/RDM).

House style throughout (figstyle), annotated. Auto-discovers which models have
a trace (prefers the _long run), builds the model from its spec to run PPCs.

    ~/mambaforge/envs/bauer/bin/python make_report.py --what params
    ~/mambaforge/envs/bauer/bin/python make_report.py --what ppc --n-draws 100
"""
import argparse
import os.path as op

import arviz as az
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import figstyle
from fit_ddm import build_model
from bauer.utils.bayes import softplus

# tag -> (spec for build_model, arch, pretty title)
REGISTRY = [
    ("choice_monly", "choice:memory_noise_sd", "choice", "Choice (σ memory ~ group)"),
    ("choice_ponly", "choice:perceptual_noise_sd", "choice", "Choice (σ perceptual ~ group)"),
    ("choice_group", "choice_group", "choice", "Choice (noise ~ group)"),
    ("choice_full", "choice_full", "choice", "Choice + prior ~ group"),
    ("ddm_monly", "ddm:memory_noise_sd", "ddm", "DDM (σ memory ~ group)"),
    ("ddm_ponly", "ddm:perceptual_noise_sd", "ddm", "DDM (σ perceptual ~ group)"),
    ("ddm_pm", "ddm:perceptual_noise_sd,memory_noise_sd", "ddm", "DDM (noise ~ group)"),
    ("ddm_group", "ddm_group", "ddm", "DDM (noise + a ~ group)"),
    ("ddm_pmt0", "ddm:perceptual_noise_sd,memory_noise_sd,t0", "ddm", "DDM (noise + t0 ~ group)"),
    ("ddm_t0group", "ddm_t0group", "ddm", "DDM (noise + a + t0 ~ group)"),
    ("rdm_monly", "rdm:memory_noise_sd", "rdm", "RDM (σ memory ~ group)"),
    ("rdm_ponly", "rdm:perceptual_noise_sd", "rdm", "RDM (σ perceptual ~ group)"),
    ("rdm_pm", "rdm:perceptual_noise_sd,memory_noise_sd", "rdm", "RDM (noise ~ group)"),
    ("rdm_group", "rdm_group", "rdm", "RDM (noise + a ~ group)"),
    ("rdm_pmt0", "rdm:perceptual_noise_sd,memory_noise_sd,t0", "rdm", "RDM (noise + t0 ~ group)"),
    ("rdm_pmat0", "rdm:perceptual_noise_sd,memory_noise_sd,a,t0", "rdm", "RDM (noise + a + t0 ~ group)"),
]
LABELS = {"perceptual_noise_sd": "σ perceptual", "memory_noise_sd": "σ memory",
          "a": "Threshold a", "t0": "Non-decision t₀", "w_0": "Drift w0",
          "w_d": "Drift wd", "w_s": "Drift ws", "prior_mu": "Prior μₚ",
          "prior_sd": "Prior σₚ"}
QUANT = [0.1, 0.3, 0.5, 0.7, 0.9]
QCOL = None  # set after figstyle


def trace_path(tag):
    for p in (f"results/model-{tag}_long_trace.nc", f"results/model-{tag}_trace.nc"):
        if op.exists(p):
            return p
    return None


def load_df(unique_idx=False):
    df = pd.read_csv("data/magjudge_behavior_DNumRisk.csv")
    df = df[df["rt"] >= 0.20].copy()
    df["choice"] = df["choice"].astype(bool)
    df["chose_n2"] = df["chose_n2"].astype(bool)
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    return df.set_index(["subject", "run", "trial_nr"]) if unique_idx else df.set_index("subject")


def _tf(name, x):
    pos = (name.endswith("_sd") or "noise" in name or name in ("a", "t0")
           or name.startswith("w_"))
    return softplus(x) if pos else x


# ---------------- params page ----------------
def params_page(pdf, tag, title):
    import matplotlib.pyplot as plt
    post = az.from_netcdf(trace_path(tag)).posterior
    order = ["perceptual_noise_sd", "memory_noise_sd", "a", "t0", "w_0", "w_d",
             "w_s", "prior_mu", "prior_sd"]
    vars_ = [p for p in order if f"{p}_mu" in post]
    n = len(vars_)
    fig, axes = plt.subplots(1, n, figsize=(1.55 * n + 0.6, 2.8))
    axes = np.atleast_1d(axes).ravel()
    any_flagged = False
    for ax, p in zip(axes, vars_):
        v = f"{p}_mu"
        rd = next(d for d in post[v].dims if d.endswith("_regressors"))
        gl = [l for l in post.coords[rd].values if l != "Intercept"]
        icpt = post[v].sel({rd: "Intercept"}).values.ravel()
        series = [("control", _tf(p, icpt), figstyle.GROUP["control"])]
        ess = np.inf
        if gl:
            gd = post[v].sel({rd: gl[0]})
            series.append(("dyscalc", _tf(p, icpt + gd.values.ravel()),
                           figstyle.GROUP["dyscalc"]))
            ess = float(az.ess(gd.transpose("chain", "draw").values))
        else:
            series = [("", _tf(p, icpt), figstyle.NEUTRAL)]
        for x, (lab, draws, col) in enumerate(series):
            lo, hi = az.hdi(draws, hdi_prob=0.94)
            ax.plot([x, x], [lo, hi], color=col, lw=2.4, solid_capstyle="round")
            ax.scatter([x], [draws.mean()], color=col, s=24, zorder=3)
        ax.set_xticks(range(len(series)))
        ax.set_xticklabels([s[0][:4].capitalize() for s in series], fontsize=7)
        ax.set_xlim(-0.6, len(series) - 0.4)
        ax.set_xlabel(LABELS.get(p, p), fontsize=8)
        if gl:
            d, c = series[1][1], series[0][1]
            flagged = ess < 100
            any_flagged = any_flagged or flagged
            flag = " *" if flagged else ""
            ax.annotate(f"P={ (d>c).mean():.2f}{flag}", xy=(0.5, 0.99),
                        xycoords="axes fraction", ha="center", va="top",
                        fontsize=6.5, color="0.35")
    axes[0].set_ylabel("Posterior (nat. units)")
    suffix = "   (* = ESS<100, not converged)" if any_flagged else ""
    fig.suptitle(title + suffix, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    import seaborn as sns
    sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig); plt.close(fig)


# ---------------- ppc page ----------------
def get_cache(tag, spec, n_draws):
    cache = f"results/ppc_{tag}.parquet"
    if op.exists(cache):
        return pd.read_parquet(cache)
    df = load_df(unique_idx=True)
    model = build_model(spec, df)
    if "choice" in spec:                 # static model's ppc needs the estimation model built
        model.build_estimation_model(data=df, hierarchical=True)
    idata = az.from_netcdf(trace_path(tag))
    ppc = model.ppc(df, idata, n_posterior_samples=n_draws, progressbar=False)
    ppc = ppc.reset_index().merge(
        df[["n1", "n2", "group", "chose_n2"]].reset_index(),
        on=["subject", "run", "trial_nr"], how="left")
    ppc.to_parquet(cache)
    return ppc


def ppc_page(pdf, tag, spec, arch, title, n_draws):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ppc = get_cache(tag, spec, n_draws)
    has_rt = "simulated_rt" in ppc.columns and arch in ("ddm", "rdm")
    obs = load_df().reset_index()
    npan = 2 + (1 if has_rt else 0)        # n1 ; n1xorder interaction ; [QPP]
    fig, axes = plt.subplots(1, npan, figsize=(4.4 * npan, 3.6), squeeze=False)
    axes = axes.ravel()
    # --- panel 1: P(chose n2) vs n1 ---
    ax = axes[0]
    for g in ["control", "dyscalc"]:
        o = (obs[obs.group == g].groupby("n1")["chose_n2"].mean())
        ax.scatter(o.index, o.values, color=figstyle.GROUP[g], s=30, zorder=3)
        ps = ppc[ppc.group == g]
        per = ps.groupby(["n1", "ppc_sample"])["simulated_choice"].mean().reset_index()
        agg = per.groupby("n1")["simulated_choice"].agg(
            m="mean", lo=lambda x: az.hdi(x.values, 0.94)[0],
            hi=lambda x: az.hdi(x.values, 0.94)[1])
        ax.plot(agg.index, agg["m"], color=figstyle.GROUP[g], lw=1.8)
        ax.fill_between(agg.index, agg.lo, agg.hi, color=figstyle.GROUP[g], alpha=.2)
    ax.axhline(0.5, ls=":", c="0.6", lw=0.7)
    ax.set_xlabel("n1 (reference)"); ax.set_ylabel("P(chose n2)")
    ax.annotate("Points = data\nBand = model", xy=(0.04, 0.96),
                xycoords="axes fraction", fontsize=6.5, color="0.35", va="top")
    # --- panel 2: n1 x order interaction — psychometric vs log(n2/n1) per n1 ---
    ax = axes[1]
    obs2 = obs.copy(); obs2["lr"] = np.log(obs2["n2"] / obs2["n1"])
    ppc2 = ppc.copy(); ppc2["lr"] = np.log(ppc2["n2"] / ppc2["n1"])
    n1s = sorted(obs2["n1"].unique())
    cmap = sns.color_palette("viridis", as_cmap=True)
    for j, n1v in enumerate(n1s):
        col = cmap(j / max(1, len(n1s) - 1))
        od = obs2[obs2.n1 == n1v].groupby("lr")["chose_n2"].mean()
        ax.scatter(od.index, od.values, color=col, s=18, zorder=3)
        md = ppc2[ppc2.n1 == n1v].groupby("lr")["simulated_choice"].mean()
        ax.plot(md.index, md.values, color=col, lw=1.6, zorder=2)
        # label the n1 level at the rightmost point
        ax.annotate(f"n1={int(n1v)}", xy=(md.index.max(), md.iloc[-1]),
                    xytext=(3, 0), textcoords="offset points", fontsize=6,
                    color=col, va="center")
    ax.axhline(0.5, ls=":", c="0.6", lw=0.7); ax.axvline(0, ls=":", c="0.6", lw=0.7)
    ax.set_xlabel("Log(n2/n1)"); ax.set_ylabel("P(chose n2)")
    ax.set_title("n1 × order interaction", fontsize=9)
    ax.annotate("Curves shift\nwith n1 = order effect", xy=(0.04, 0.04),
                xycoords="axes fraction", fontsize=6, color="0.35", va="bottom")
    # --- panel 3: QPP (RT models) ---
    if has_rt:
        ax = axes[2]
        ppc = ppc.copy()
        ppc["correct"] = ppc["simulated_choice"].astype(bool) == (ppc["n2"] > ppc["n1"])
        ppc["absL"] = np.abs(np.log(ppc["n2"] / ppc["n1"]))
        obs2 = obs.copy()
        obs2["correct"] = obs2["chose_n2"] == (obs2["n2"] > obs2["n1"])
        obs2["absL"] = np.abs(np.log(obs2["n2"] / obs2["n1"]))
        edges = np.unique(np.quantile(obs2["absL"], np.linspace(0, 1, 6)))

        def qtab(d, rtc, by_s):
            d = d.copy(); d["b"] = pd.cut(d["absL"], edges, include_lowest=True)
            keys = ["b"] + (["ppc_sample"] if by_s else [])
            rows = []
            for k, sub in d.groupby(keys, observed=True):
                if len(sub) < 5:
                    continue
                p = sub["correct"].mean()
                for isc, x in [(True, p), (False, 1 - p)]:
                    r = sub.loc[sub["correct"] == isc, rtc]
                    if len(r) < 5:
                        continue
                    for qi, q in enumerate(QUANT):
                        rows.append(dict(qi=qi, x=x, rt=r.quantile(q)))
            return pd.DataFrame(rows)
        dq = qtab(obs2, "rt", False)
        mq = qtab(ppc, "simulated_rt", True)
        for qi in range(5):
            mm = mq[mq.qi == qi]
            ax.scatter(mm.x, mm.rt, s=4, color=QCOL[qi], alpha=0.05, lw=0)
            dd = dq[dq.qi == qi]
            ax.scatter(dd.x, dd.rt, s=14, color=QCOL[qi], edgecolor="k", lw=0.4)
        ax.set_xlabel("Response proportion"); ax.set_ylabel("RT quantile (s)")
        ax.set_xlim(0, 1); ax.set_ylim(0, obs2["rt"].quantile(0.99) * 1.15)
        ax.annotate("QPP: cloud=model,\npoints=data (.1–.9)", xy=(0.03, 0.04),
                    xycoords="axes fraction", fontsize=6.5, color="0.35", va="bottom")
    fig.suptitle(title, fontsize=10)
    import seaborn as sns
    fig.tight_layout(rect=(0, 0, 1, 0.95)); sns.despine(fig=fig, offset=4, trim=True)
    pdf.savefig(fig); plt.close(fig)


def main(what, n_draws):
    global QCOL
    figstyle.apply()
    import seaborn as sns
    QCOL = sns.color_palette("mako", as_cmap=True)(np.linspace(0.15, 0.85, 5))
    avail = [(t, s, a, ti) for (t, s, a, ti) in REGISTRY if trace_path(t)]
    print("models available:", [t for t, *_ in avail])

    if what in ("params", "both"):
        with PdfPages("notes/figures/report_parameters.pdf") as pdf:
            for t, s, a, ti in avail:
                params_page(pdf, t, ti)
        print("Saved notes/figures/report_parameters.pdf")
    if what in ("ppc", "both"):
        with PdfPages("notes/figures/report_ppc.pdf") as pdf:
            for t, s, a, ti in avail:
                try:
                    ppc_page(pdf, t, s, a, ti, n_draws)
                    print(f"  ppc page: {t}")
                except Exception as e:
                    print(f"  ppc FAILED {t}: {type(e).__name__}: {e}")
        print("Saved notes/figures/report_ppc.pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--what", choices=["params", "ppc", "both"], default="both")
    p.add_argument("--n-draws", type=int, default=100)
    args = p.parse_args()
    main(args.what, args.n_draws)
