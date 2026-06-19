"""The cross-model overview PDF: what choice / DDM / RDM say about the
dyscalc-control difference, in two noise decompositions, with rich PPCs.
Built on the CONVERGED cell-means traces. Reads collect_overview.py output.

  ~/mambaforge/envs/bauer/bin/python plot_overview.py
"""
import argparse, json
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import figstyle

figstyle.apply()
HERE = op.dirname(op.abspath(__file__))
GROUP = figstyle.GROUP

SHARED = ["choice", "ddm", "rdm", "rdm_max"]
N1N2 = ["choice_n1n2", "ddm_n1n2", "rdm_n1n2", "rdm_n1n2_max"]
RT_MODELS = ["ddm", "rdm"]
N1S = [5, 7, 10, 14, 20]
PARAMS_SHARED = [("perceptual_noise_sd", "Perceptual noise"),
                 ("memory_noise_sd", "Memory noise"),
                 ("a", "Threshold a"), ("t0", "Non-decision t0"),
                 ("w_0", "Baseline drift w0"), ("w_s", "Sum term ws")]
PARAMS_N1N2 = [("n1_evidence_sd", "n1 noise (held)"),
               ("n2_evidence_sd", "n2 noise (current)"),
               ("a", "Threshold a"), ("t0", "Non-decision t0"),
               ("w_0", "Baseline drift w0"), ("w_s", "Sum term ws")]


def _finish_page(fig, pdf, offset=5, trim=True):
    sns.despine(fig=fig, offset=offset, trim=trim)
    pdf.savefig(fig)
    plt.close(fig)


def page_params(pdf, S, tags, params, title, subtitle):
    tags = [t for t in tags if S.get(t, {}).get("status") == "ok"]
    params = [p for p in params
              if any(p[0] in S[t].get("contrasts", {}) for t in tags)]
    fig, axes = plt.subplots(1, len(params), figsize=(max(8, 1.8 * len(params)), 3.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    ypos = {t: i for i, t in enumerate(reversed(tags))}
    for ax, (pk, plabel) in zip(axes, params):
        for t in tags:
            c = S[t]["contrasts"].get(pk)
            if not c:
                continue
            y = ypos[t]
            credible = c["lo"] > 0 or c["hi"] < 0
            col = (("#C44E52" if c["mean"] > 0 else "#3B5BA5") if credible
                   else "#9C9C9C")
            ax.plot([c["lo"], c["hi"]], [y, y], color=col, lw=2.4,
                    solid_capstyle="round", zorder=2)
            ax.scatter([c["mean"]], [y], s=38, color=col, edgecolor="white",
                       lw=1.0, zorder=3)
            ax.annotate(f"{c['Pgt0']:.2f}", (c["mean"], y), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=6.3,
                        color=col)
        ax.axvline(0, ls=":", color="0.5", lw=0.9)
        ax.set_title(plabel, fontsize=9.5)
        ax.set_xlabel("Δ (dys − ctrl)", fontsize=8)
        ax.margins(x=0.18)
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([S[t]["pretty"] for t in ypos], fontsize=8.5)
    axes[0].set_ylim(-0.6, len(tags) - 0.4)
    fig.suptitle(title, fontsize=10.5, y=0.99)
    fig.text(0.5, 0.015, subtitle, ha="center", va="bottom", fontsize=7.8,
             style="italic", color="0.35", wrap=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.30, wspace=0.30)
    _finish_page(fig, pdf, trim=False)


def page_raw(pdf, S, tags, params, title):
    """Per-group population values in NATURAL units (effect sizes you can read
    directly), control vs dyscalc. Skips if the summary lacks 'raw' (old data)."""
    tags = [t for t in tags if S.get(t, {}).get("status") == "ok" and S[t].get("raw")]
    params = [p for p in params if any(p[0] in S[t].get("raw", {}) for t in tags)]
    if not tags or not params:
        return
    fig, axes = plt.subplots(1, len(params), figsize=(max(7, 2.0 * len(params)), 3.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    ypos = {t: i for i, t in enumerate(reversed(tags))}
    for ax, (pk, plabel) in zip(axes, params):
        for t in tags:
            r = S[t]["raw"].get(pk)
            if not r:
                continue
            y = ypos[t]
            for g, off in (("control", 0.14), ("dyscalc", -0.14)):
                d, col = r[g], GROUP[g]
                ax.plot([d["lo"], d["hi"]], [y + off, y + off], color=col, lw=2.0,
                        solid_capstyle="round", zorder=2)
                ax.scatter([d["mean"]], [y + off], s=24, color=col,
                           edgecolor="white", lw=0.8, zorder=3)
        ax.set_title(plabel, fontsize=9.5)
        ax.set_xlabel("Value (natural units)", fontsize=8)
        ax.margins(x=0.16)
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([S[t]["pretty"] for t in ypos], fontsize=8.5)
    axes[0].set_ylim(-0.6, len(tags) - 0.4)
    axes[0].annotate("Control", (0.04, 0.96), xycoords="axes fraction",
                     color=GROUP["control"], fontsize=7.5, fontweight="bold")
    axes[0].annotate("Dyscalc", (0.04, 0.88), xycoords="axes fraction",
                     color=GROUP["dyscalc"], fontsize=7.5, fontweight="bold")
    fig.suptitle(title, fontsize=10.5, y=0.99)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.85, bottom=0.16, wspace=0.30)
    _finish_page(fig, pdf, trim=False)


def page_asymmetry(pdf, S, tags):
    """sigma_n1 - sigma_n2 per group: the reliability gap that drives compression."""
    tags = [t for t in tags if S.get(t, {}).get("status") == "ok" and S[t].get("asymmetry")]
    if not tags:
        return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True)
    ypos = {t: i for i, t in enumerate(reversed(tags))}
    sp = S[tags[0]]["asymmetry"].get("sigma_p")
    # Panel A: reliability-weight dumbbells  (open = w1/n1, filled = w2/n2)
    for t in tags:
        a = S[t]["asymmetry"]; y = ypos[t]
        for g, off in (("control", 0.13), ("dyscalc", -0.13)):
            col = GROUP[g]
            w1, w2 = a["w1"][g]["mean"], a["w2"][g]["mean"]
            axA.plot([w1, w2], [y + off, y + off], color=col, lw=1.4, zorder=2)
            axA.scatter([w1], [y + off], s=30, facecolor="white", edgecolor=col,
                        lw=1.4, zorder=3)
            axA.scatter([w2], [y + off], s=30, color=col, edgecolor="white",
                        lw=0.7, zorder=3)
        axA.annotate(f"Δgap P={a['gap']['contrast']['Pgt0']:.2f}",
                     (max(a['w2']['control']['mean'], a['w2']['dyscalc']['mean']), y),
                     xytext=(8, 0), textcoords="offset points", va="center",
                     fontsize=6.5, color="0.35")
    axA.set_xlabel("Reliability weight  w = σ_p²/(σ_p²+σ_e²)", fontsize=8.5)
    axA.set_title("Reliability weights (open = n1, filled = n2)", fontsize=9.5)
    axA.margins(x=0.22)
    # Panel B: scale-invariant SD log-ratio companion
    for t in tags:
        a = S[t]["asymmetry"]["logratio"]; y = ypos[t]
        for g, off in (("control", 0.13), ("dyscalc", -0.13)):
            d, col = a[g], GROUP[g]
            axB.plot([d["lo"], d["hi"]], [y + off, y + off], color=col, lw=2.0,
                     solid_capstyle="round", zorder=2)
            axB.scatter([d["mean"]], [y + off], s=26, color=col, edgecolor="white",
                        lw=0.8, zorder=3)
        axB.annotate(f"P={a['contrast']['Pgt0']:.2f}",
                     (max(a['control']['hi'], a['dyscalc']['hi']), y),
                     xytext=(5, 0), textcoords="offset points", va="center",
                     fontsize=6.5, color="0.35")
    axB.axvline(0, ls=":", c="0.5", lw=0.9)
    axB.set_xlabel("log(σ_n1 / σ_n2)   (scale-invariant)", fontsize=8.5)
    axB.set_title("SD log-ratio (companion)", fontsize=9.5)
    axB.margins(x=0.18)
    axA.set_yticks(list(ypos.values()))
    axA.set_yticklabels([S[t]["pretty"] for t in ypos], fontsize=9)
    axA.set_ylim(-0.6, len(tags) - 0.4)
    axA.annotate("Control", (0.04, 0.95), xycoords="axes fraction",
                 color=GROUP["control"], fontsize=8, fontweight="bold")
    axA.annotate("Dyscalc", (0.04, 0.87), xycoords="axes fraction",
                 color=GROUP["dyscalc"], fontsize=8, fontweight="bold")
    fig.suptitle("Compression driver — reliability weights"
                 + (f"  (σ_p={sp:.2f})" if sp else ""), fontsize=10.5, y=1.0)
    fig.text(0.5, 0.02, "Lower weight = more regression to the prior = more compression. "
             "n1 (open) sits below n2 (filled) -> n1 compressed more. In dyscalc BOTH "
             "weights drop (more noise) AND the gap w2−w1 widens nonlinearly -> more "
             "differential compression. Δgap P = P(gap_dys > gap_ctrl).",
             ha="center", va="bottom", fontsize=7.3, style="italic", color="0.35",
             wrap=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.86, bottom=0.26, wspace=0.12)
    _finish_page(fig, pdf, trim=False)


def page_group_sds(pdf, S, tags, params, title):
    tags = [t for t in tags if S.get(t, {}).get("status") == "ok"
            and S[t].get("group_sds")]
    fig, axes = plt.subplots(1, len(params), figsize=(max(8, 1.8 * len(params)), 3.4),
                             sharey=True, squeeze=False)
    axes = axes[0]
    ypos = {t: i for i, t in enumerate(reversed(tags))}
    for ax, (pk, plabel) in zip(axes, params):
        for t in tags:
            gs = S[t]["group_sds"].get(pk)
            if not gs:
                continue
            y = ypos[t]
            for g, off in (("control", 0.14), ("dyscalc", -0.14)):
                d, col = gs[g], GROUP[g]
                ax.plot([d["lo"], d["hi"]], [y + off, y + off], color=col, lw=2.0,
                        solid_capstyle="round", zorder=2)
                ax.scatter([d["mean"]], [y + off], s=24, color=col,
                           edgecolor="white", lw=0.8, zorder=3)
            ax.annotate(f"{gs['contrast']['Pgt0']:.2f}",
                        (max(gs['control']['hi'], gs['dyscalc']['hi']), y),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=6.3, color="0.35")
        ax.set_title(plabel, fontsize=9.5)
        ax.set_xlabel("Between-subject SD τ", fontsize=8)
        ax.margins(x=0.20)
    axes[0].set_yticks(list(ypos.values()))
    axes[0].set_yticklabels([S[t]["pretty"] for t in ypos], fontsize=8.5)
    axes[0].set_ylim(-0.6, len(tags) - 0.4)
    axes[0].annotate("Control", (0.04, 0.96), xycoords="axes fraction",
                     color=GROUP["control"], fontsize=7.5, fontweight="bold")
    axes[0].annotate("Dyscalc", (0.04, 0.88), xycoords="axes fraction",
                     color=GROUP["dyscalc"], fontsize=7.5, fontweight="bold")
    fig.suptitle(title, fontsize=10.5, y=0.99)
    fig.text(0.5, 0.015, "Per-group between-subject SD (heterogeneity). "
             "number = P(τ_dyscalc > τ_control).", ha="center", va="bottom",
             fontsize=7.8, style="italic", color="0.35")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.26, wspace=0.30)
    _finish_page(fig, pdf, trim=False)


def _curve_panel(ax, Z, key, ykey, obskey, log_x):
    for g, col in GROUP.items():
        k = f"{key}__{g}"
        if f"{k}__x" in Z.files:
            x = Z[f"{k}__x"]; o = np.argsort(x); x = x[o]
            ax.fill_between(x, Z[f"{k}__lo"][o], Z[f"{k}__hi"][o], color=col,
                            alpha=0.22, lw=0)
            ax.plot(x, Z[f"{k}__med"][o], color=col, lw=1.2, zorder=3)
        ok = f"{obskey}__{g}"
        if f"{ok}__x" in Z.files:
            ox = Z[f"{ok}__x"]; oo = np.argsort(ox)
            ax.scatter(ox[oo], Z[f"{ok}__{ykey}"][oo], s=12, facecolor="white",
                       edgecolor=col, lw=1.0, zorder=4)


def page_by_n1(pdf, S, Z, kind, tags=("ddm", "rdm")):
    """kind='psy' -> P(chose n2) vs log-ratio; kind='rtpsy' -> median RT."""
    tags = [t for t in tags if S.get(t, {}).get("status") == "ok"]
    ylab = "P(chose n2)" if kind == "psy" else "Median RT (s)"
    ykey = "p" if kind == "psy" else "y"
    fig, axes = plt.subplots(len(tags), len(N1S), figsize=(8.6, 2.5 * len(tags) + 0.4),
                             sharey="row", sharex=True, squeeze=False)
    for r, tag in enumerate(tags):
        for c, n1 in enumerate(N1S):
            ax = axes[r][c]
            _curve_panel(ax, Z, f"{tag}__{kind}__n1{n1}", ykey,
                         f"obs__{kind}__n1{n1}" if kind == "psy"
                         else f"obs__rtpsy__n1{n1}", log_x=True)
            ax.axvline(0, ls=":", c="0.6", lw=0.8)
            if kind == "psy":
                ax.axhline(0.5, ls=":", c="0.6", lw=0.8); ax.set_ylim(0, 1)
            if r == 0:
                ax.set_title(f"n1 = {n1}", fontsize=9)
            if r == len(tags) - 1:
                ax.set_xlabel("log n2 − log n1", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{S[tag]['pretty']}\n{ylab}", fontsize=8.5)
    axes[0][0].annotate("Control", (0.05, 0.9), xycoords="axes fraction",
                        color=GROUP["control"], fontsize=7, fontweight="bold")
    axes[0][0].annotate("Dyscalc", (0.05, 0.78), xycoords="axes fraction",
                        color=GROUP["dyscalc"], fontsize=7, fontweight="bold")
    what = ("Psychometric curves" if kind == "psy" else "RT")
    fig.suptitle(f"{what} by reference magnitude n1 — DDM & RDM PPC "
                 "(points = data, band = model 94% HDI)", fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _finish_page(fig, pdf)


def page_ppc(pdf, S, Z):
    models = [t for t in RT_MODELS if S.get(t, {}).get("status") == "ok"]
    rows = [("compression", "P(chose n2)", "n1", "n1"),
            ("difficulty", "Median RT (s)", "|log(n2/n1)|  (difficulty)", "x"),
            ("magnitude", "Median RT (s)", "n1 + n2  (magnitude / stake)", "x")]
    fig, axes = plt.subplots(len(rows), len(models), figsize=(7.0, 7.0), squeeze=False)
    for i, (kind, ylab, xlab, _) in enumerate(rows):
        for j, t in enumerate(models):
            ax = axes[i][j]
            if kind == "compression":
                for g, col in GROUP.items():
                    k = f"{t}__{g}"
                    if f"{k}__n1" in Z.files:
                        x = Z[f"{k}__n1"]; o = np.argsort(x); x = x[o]
                        ax.fill_between(x, Z[f"{k}__lo"][o], Z[f"{k}__hi"][o],
                                        color=col, alpha=0.22, lw=0)
                        ax.plot(x, Z[f"{k}__med"][o], color=col, lw=1.2, zorder=3)
                    ox = Z[f"obs__{g}__n1"]; oo = np.argsort(ox)
                    ax.scatter(ox[oo], Z[f"obs__{g}__p"][oo], s=13,
                               facecolor="white", edgecolor=col, lw=1.0, zorder=4)
                ax.axhline(0.5, ls=":", c="0.6", lw=0.8); ax.set_ylim(0.25, 0.7)
            else:
                _curve_panel(ax, Z, f"{t}__{kind}", "y", f"obsrt__{kind}", True)
            if i == 0:
                ax.set_title(S[t]["pretty"], fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9)
            ax.set_xlabel(xlab, fontsize=8)
    axes[0][0].annotate("Control", (0.04, 0.9), xycoords="axes fraction",
                        color=GROUP["control"], fontsize=7.5, fontweight="bold")
    axes[0][0].annotate("Dyscalc", (0.04, 0.8), xycoords="axes fraction",
                        color=GROUP["dyscalc"], fontsize=7.5, fontweight="bold")
    fig.suptitle("PPCs — points = data, band = model 94% HDI", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _finish_page(fig, pdf)


def page_loo(pdf, S):
    rt = [t for t in S if S[t].get("family") == "rt" and "elpd_loo" in S[t]]
    rt.sort(key=lambda t: -S[t]["elpd_loo"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if rt:
        best = max(S[t]["elpd_loo"] for t in rt)
        ys = np.arange(len(rt))[::-1]
        for y, t in zip(ys, rt):
            d = S[t]
            delta = d["elpd_loo"] - best
            ax.errorbar(delta, y, xerr=d.get("elpd_se", 0), fmt="o", ms=7,
                        color="#3B5BA5", capsize=3, lw=1.2)
            bk = d.get("bad_k", 0)
            flag = " [LOO unreliable]" if bk > 0.005 * d.get("n_obs", 1e9) else ""
            dec = "n1/n2" if "n1n2" in t else "perc/mem"
            ax.annotate(f"{d['pretty']} ({dec}): elpd {d['elpd_loo']:.0f} "
                        f"(Δ{delta:+.1f}), bad-k {bk}{flag}",
                        (delta, y), xytext=(10, 0), textcoords="offset points",
                        va="center", fontsize=8)
        ax.axvline(0, ls=":", color="0.5", lw=0.9)
        ax.set_yticks(ys); ax.set_yticklabels([])
        ax.set_xlabel("Δ elpd_loo vs best  (choice + RT models; higher = better)")
        ax.set_ylim(-0.6, len(rt) - 0.4); ax.margins(x=0.28)
    ax.set_title("Model comparison (LOO)", fontsize=10)
    note = ("Static choice models are off this axis (choice-only likelihood). "
            "Among choice+RT models the RDM dominates the DDM, with reliable LOO; "
            "the noise decomposition barely moves fit — a reparameterization, not "
            "a better model.")
    fig.text(0.5, 0.04, note, ha="center", va="bottom", fontsize=8, color="0.35",
             wrap=True)
    fig.subplots_adjust(bottom=0.30, top=0.9, left=0.04, right=0.97)
    _finish_page(fig, pdf, trim=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="notes/data/overview.npz")
    p.add_argument("--json", default="notes/data/overview.json")
    p.add_argument("--out", default="notes/figures/overview_verdict.pdf")
    a = p.parse_args()
    S = json.load(open(op.join(HERE, a.json)))
    for v in S.values():
        if isinstance(v, dict) and "pretty" in v:
            v["pretty"] = v["pretty"].replace("≡", "=").replace("RDM (full drift)",
                                                                "RDM (+w0/ws)")
    Z = np.load(op.join(HERE, a.npz), allow_pickle=True)
    with PdfPages(op.join(HERE, a.out)) as pdf:
        page_params(pdf, S, SHARED, PARAMS_SHARED,
                    "Where the deficit sits — shared (perceptual / memory) decomposition",
                    "Accumulators load it on perceptual noise, the choice model on memory. "
                    "With a/t0 also freed, some deficit redistributes onto threshold (RDM) "
                    "and t0 (DDM) — they trade off, so that split is not cleanly identified. "
                    "Gray = 94% HDI includes 0; number = P(Δ>0).")
        page_params(pdf, S, N1N2, PARAMS_N1N2,
                    "Where the deficit sits — independent n1/n2 decomposition",
                    "n1 = held (remembered) first number, n2 = current second number. "
                    "Gray = 94% HDI includes 0; number = P(Δ>0).")
        page_raw(pdf, S, N1N2, [("n1_evidence_sd", "n1 noise (held)"),
                                ("n2_evidence_sd", "n2 noise (current)")],
                 "Raw effect sizes — n1 / n2 encoding SD per group (natural units)")
        page_asymmetry(pdf, S, N1N2)
        page_raw(pdf, S, SHARED, [("perceptual_noise_sd", "Perceptual noise"),
                                  ("memory_noise_sd", "Memory noise")],
                 "Raw effect sizes — perceptual / memory SD per group (natural units)")
        page_group_sds(pdf, S, SHARED, PARAMS_SHARED[:4],
                       "Between-subject heterogeneity — per-group SDs (τ) by parameter")
        page_by_n1(pdf, S, Z, "psy", tags=("ddm", "rdm"))
        page_by_n1(pdf, S, Z, "rtpsy", tags=("ddm", "rdm"))
        page_ppc(pdf, S, Z)
        page_loo(pdf, S)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
