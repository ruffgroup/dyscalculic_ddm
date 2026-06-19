"""PSIS-LOO model comparison, by likelihood family.

LOO is only comparable WITHIN a likelihood family (same observations):
  * RT family  -> DDM / RDM (rt+choice).  DDM vs RDM is fair (same data).
  * Choice family -> static choice (choice-only Bernoulli). raw vs KLW are
    comparable; NOT comparable to the RT family.

Within each family we walk the ladder: nothing (null) -> only perceptual ->
only memory -> both noises -> everything (all free), plus the n1/n2-separate
parameterization. loo is computed one model at a time (memory-safe), then
ranked by elpd; higher elpd = better.

  ~/mambaforge/envs/bauer/bin/python make_key_loo.py
"""
import gc
import os.path as op
import warnings

import arviz as az
import numpy as np
import matplotlib.pyplot as plt

import figstyle

warnings.filterwarnings("ignore")
figstyle.apply()
HERE = op.dirname(op.abspath(__file__))

# (label, sub-family colour key, trace basename)
RT = [
    ("DDM null",        "ddm", "model-ddm_null_long_trace.nc"),
    ("DDM σperc",       "ddm", "model-ddm_ponly_long_trace.nc"),
    ("DDM σmem",        "ddm", "model-ddm_monly_long_trace.nc"),
    ("DDM both",        "ddm", "model-ddm_pm_long_trace.nc"),
    ("DDM both+a",      "ddm", "model-ddm_group_long_trace.nc"),
    ("DDM all free",    "ddm", "model-ddm_allfree_long_trace.nc"),
    ("DDM n1/n2",       "ddm", "model-ddm_n1n2_long_trace.nc"),
    ("RDM null",        "rdm", "model-rdm_null_long_trace.nc"),
    ("RDM σperc",       "rdm", "model-rdm_ponly_long_trace.nc"),
    ("RDM σmem",        "rdm", "model-rdm_monly_long_trace.nc"),
    ("RDM both",        "rdm", "model-rdm_pm_long_trace.nc"),
    ("RDM both+a",      "rdm", "model-rdm_group_long_trace.nc"),
    ("RDM all free",    "rdm", "model-rdm_allfree_long_trace.nc"),
    ("RDM n1/n2",       "rdm", "model-rdm_n1n2_long_trace.nc"),
    ("RDM $-w_s$ σperc", "nows", "model-rdm_nows_ponly_long_trace.nc"),
    ("RDM $-w_s$ σmem",  "nows", "model-rdm_nows_monly_long_trace.nc"),
    ("RDM $-w_s$ both",  "nows", "model-rdm_nows_pm_long_trace.nc"),
    ("RDM $-w_s$ n1/n2", "nows", "model-rdm_nows_n1n2_long_trace.nc"),
    ("RDM $w_d{=}1$ null",  "nowd", "model-rdm_nowd_null_long_trace.nc"),
    ("RDM $w_d{=}1$ σperc", "nowd", "model-rdm_nowd_ponly_long_trace.nc"),
    ("RDM $w_d{=}1$ σmem",  "nowd", "model-rdm_nowd_monly_long_trace.nc"),
    ("RDM $w_d{=}1$ both",  "nowd", "model-rdm_nowd_pm_long_trace.nc"),
    ("RDM $w_d{=}1$ n1/n2", "nowd", "model-rdm_nowd_n1n2_long_trace.nc"),
]
CHOICE = [
    ("null (raw)",      "raw", "model-choice_null_long_trace.nc"),
    ("σperc (raw)",     "raw", "model-choice_ponly_long_trace.nc"),
    ("σmem (raw)",      "raw", "model-choice_monly_long_trace.nc"),
    ("both (raw)",      "raw", "model-choice_group_long_trace.nc"),
    ("+prior (raw)",    "raw", "model-choice_full_trace.nc"),
    ("n1/n2 (raw)",     "raw", "model-choice_n1n2_long_trace.nc"),
    ("null (KLW)",      "klw", "model-choice_null_klw_trace.nc"),
    ("σperc (KLW)",     "klw", "model-choice_ponly_klw_trace.nc"),
    ("σmem (KLW)",      "klw", "model-choice_monly_klw_trace.nc"),
    ("both (KLW)",      "klw", "model-choice_group_klw_trace.nc"),
    ("+prior (KLW)",    "klw", "model-choice_full_klw_trace.nc"),
    ("n1/n2 (KLW)",     "klw", "model-choice_n1n2_klw_trace.nc"),
]
COL = {"ddm": "#3B5BA5", "rdm": "#C44E52", "nows": "#E0936F", "nowd": "#7A4FA3",
       "raw": "#B0B0B0", "klw": "#555555"}


def loo_of(basename):
    p = op.join(HERE, "results", basename)
    if not op.exists(p):
        return None
    idata = az.from_netcdf(p)
    if "log_likelihood" not in idata.groups():
        return None
    r = az.loo(idata, pointwise=True)
    # PSIS reliability: fraction of trials with Pareto k > 0.7 (loo untrustworthy
    # when this is non-trivial). p_loo ≫ true param count is the same red flag.
    k = np.asarray(r.pareto_k.values) if hasattr(r, "pareto_k") else np.array([])
    frac_bad = float((k > 0.7).mean()) if k.size else np.nan
    out = (float(r.elpd_loo), float(r.se), float(getattr(r, "p_loo", np.nan)),
           frac_bad)
    del idata, r
    gc.collect()
    return out


def compute(members):
    rows = []
    for label, fam, base in members:
        res = loo_of(base)
        if res is None:
            print(f"   (skip, missing/no-LL) {label}")
            continue
        elpd, se, ploo, fbad = res
        rows.append(dict(label=label, fam=fam, elpd=elpd, se=se, p_loo=ploo,
                         frac_bad=fbad))
        flag = "  ⚠PSIS" if (fbad > 0.05 or ploo > 1000) else ""
        print(f"   {label:16s} elpd={elpd:10.1f} ± {se:5.1f}  p_loo={ploo:7.1f}"
              f"  bad_k={fbad:.2f}{flag}")
    rows.sort(key=lambda d: d["elpd"])              # worst -> best (bottom->top)
    best = rows[-1]["elpd"] if rows else 0.0
    for d in rows:
        d["delpd"] = d["elpd"] - best
    return rows


def _bad(d):
    return (d.get("frac_bad", 0) > 0.05) or (d["p_loo"] > 1000)


def panel(ax, rows, title):
    # autoscale to the RELIABLE models so their differences stay legible;
    # unreliable (X) models are clamped to the left edge with their true Δ.
    rel = [d for d in rows if not _bad(d)]
    if rel:
        lo = min(d["elpd"] - d["se"] for d in rel)
        hi = max(d["elpd"] + d["se"] for d in rel)
    else:
        lo = min(d["elpd"] for d in rows); hi = max(d["elpd"] for d in rows)
    span = max(hi - lo, 1.0)
    xlo, xhi = lo - 0.30 * span, hi + 0.10 * span
    for i, d in enumerate(rows):
        c = COL[d["fam"]]
        bad = _bad(d)
        clamped = bad and d["elpd"] < xlo
        xp = xlo if clamped else d["elpd"]
        ax.errorbar(xp, i, xerr=(0 if clamped else d["se"]),
                    fmt="X" if bad else "o", ms=6 if bad else 5, color=c,
                    ecolor=c, elinewidth=1.2, capsize=2,
                    mfc=(c if bad else "white"), mec=c, mew=1.3)
        if clamped:
            tag = f"◀ Δ{d['delpd']:.0f}  ⚠p_loo={d['p_loo']:.0f}"
        else:
            tag = f"Δ{d['delpd']:.0f}" if d["delpd"] < 0 else ""
            if bad:
                tag += f"  ⚠p_loo={d['p_loo']:.0f}"
        if tag:
            ax.annotate(tag.strip(), (xp, i), (7, 0), textcoords="offset points",
                        va="center", fontsize=6.2,
                        color=("#B23B3B" if bad else "0.45"))
    ax.set_xlim(xlo - 0.04 * span, xhi)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([d["label"] for d in rows], fontsize=7.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("elpd_loo  (higher = better; Δ vs best)")
    ax.set_title(title, fontsize=9.5)


def main():
    print("[RT family]");   rt = compute(RT)
    print("[Choice family]"); ch = compute(CHOICE)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 0.32 * max(len(rt), len(ch)) + 1.4),
                             constrained_layout=True)
    panel(axes[0], rt, "RT family (DDM / RDM)")
    panel(axes[1], ch, "Choice family (raw / KLW)")
    fig.suptitle("PSIS-LOO ladder: null, σperc, σmem, both, all free (+ n1/n2). "
                 "X = unreliable (Pareto-k/p_loo)", fontsize=9.5)
    figstyle.finish(fig, op.join(HERE, "notes", "figures", "key_models_loo.pdf"),
                    trim=False)


if __name__ == "__main__":
    main()
