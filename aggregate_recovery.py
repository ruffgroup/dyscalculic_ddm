"""Aggregate parametric-bootstrap recovery reps into a per-model recovery cloud.

Reads results/recover_boot/<model>_rep*.csv and, per model class, plots
recovered-vs-true for each noise contrast (the diagonal = identity = perfect
recovery) plus the perceptual<->memory cross-loading. Quantifies: recovery
slope, Pearson r, bias, and the cross-loading r (true of one component vs
recovered of the other) — the direct identifiability readout.

  python aggregate_recovery.py --models choice ddm ddm_sv
"""
import argparse
import glob
import os.path as op

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figstyle

figstyle.apply()
HERE = op.dirname(op.abspath(__file__))
LAB = {"perceptual_noise_sd": "Perceptual", "memory_noise_sd": "Memory"}
SHORT = {"perceptual_noise_sd": "Perc.", "memory_noise_sd": "Mem."}
PRETTY = {"choice": "Choice (raw)", "choice_klw": "Choice (KLW)",
          "ddm": "DDM", "ddm_sv": "sv-DDM"}
COL = {"perceptual_noise_sd": "#3B8C8C", "memory_noise_sd": "#D98A29"}


def load(model):
    files = sorted(glob.glob(op.join(HERE, "results", "recover_boot", f"{model}_rep*.csv")))
    if not files:
        return None
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def stats(t, r):
    m = np.isfinite(t) & np.isfinite(r)
    t, r = t[m], r[m]
    if len(t) < 3:
        return dict(n=len(t), r=np.nan, slope=np.nan, bias=np.nan)
    slope, intercept = np.polyfit(t, r, 1)
    return dict(n=len(t), r=np.corrcoef(t, r)[0, 1], slope=slope,
               bias=float(np.mean(r - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["choice", "ddm", "ddm_sv"])
    ap.add_argument("--rhat-max", type=float, default=1.05,
                    help="drop reps whose worst group r̂ exceeds this")
    args = ap.parse_args()

    data = {m: load(m) for m in args.models}
    data = {m: d for m, d in data.items() if d is not None and len(d)}
    if not data:
        print("No recovery results found yet.")
        return

    # Convergence gate: a bootstrap rep that didn't mix carries no information
    # about recoverability, only sampler noise. Keep the raw count to report
    # how many survived (DDM reps used short chains and mostly fail this).
    n_raw = {m: len(d) for m, d in data.items()}
    data = {m: d[d["max_rhat"] <= args.rhat_max].reset_index(drop=True)
            for m, d in data.items()}

    # One 2x2 cross-recovery matrix per model: rows = recovered component,
    # cols = TRUE component. Diagonal = self-recovery (want identity line);
    # off-diagonal = leakage (want flat, zero-centered).
    comps = ["perceptual_noise_sd", "memory_noise_sd"]
    nm = len(data)
    fig, axes = plt.subplots(2 * nm, 2, figsize=(5.6, 3.0 * nm),
                             constrained_layout=True, squeeze=False)
    for mi, (model, d) in enumerate(data.items()):
        # shared symmetric limit across the whole 2x2 block for comparability
        allv = np.concatenate([d[f"true_{c}"].values for c in comps]
                              + [d[f"rec_{c}"].values for c in comps])
        lim = np.nanmax(np.abs(allv)) * 1.1
        for ri, rec in enumerate(comps):          # row = recovered component
            for ci, tru in enumerate(comps):      # col = true component
                ax = axes[2 * mi + ri, ci]
                t, r = d[f"true_{tru}"].values, d[f"rec_{rec}"].values
                s = stats(t, r)
                diag = (ri == ci)
                ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)
                if diag:
                    ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.6, ls=":")
                    col = COL[rec]
                    txt = f"r={s['r']:.2f}\nslope={s['slope']:.2f}\nbias={s['bias']:+.2f}"
                else:
                    col = "0.45"
                    txt = f"leak r={s['r']:.2f}\nslope={s['slope']:.2f}"
                ax.scatter(t, r, s=13, color=col, alpha=0.55, edgecolor="none")
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                ax.set_aspect("equal")
                ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top",
                        fontsize=6.8, color=("0.15" if diag else "0.45"))
                # outer labels only
                if ri == 1:
                    ax.set_xlabel(f"True {SHORT[tru]} Δ")
                if ci == 0:
                    ax.set_ylabel(f"Rec. {SHORT[rec]} Δ")
        # model label + convergence count as a left-aligned section header on
        # the block's top-left axis (set_title -> constrained_layout reserves
        # space, so it never collides with the block above).
        axes[2 * mi, 0].set_title(
            f"{PRETTY.get(model, model)}   (n={len(d)}/{n_raw[model]} converged, "
            f"median acc={d['sim_acc'].median():.2f})",
            loc="left", fontsize=8.5, fontweight="bold", pad=8)

        # console summary
        print(f"\n[{model}]  n={len(d)}/{n_raw[model]} converged (r̂≤{args.rhat_max})  "
              f"median sim_acc={d['sim_acc'].median():.3f}")
        for p in comps:
            s = stats(d[f"true_{p}"].values, d[f"rec_{p}"].values)
            print(f"   self {LAB[p]:11s}: r={s['r']:.3f} slope={s['slope']:.2f} bias={s['bias']:+.3f}")
        lk1 = stats(d["true_memory_noise_sd"].values, d["rec_perceptual_noise_sd"].values)
        lk2 = stats(d["true_perceptual_noise_sd"].values, d["rec_memory_noise_sd"].values)
        print(f"   leak rec_perc~true_mem r={lk1['r']:.3f}  rec_mem~true_perc r={lk2['r']:.3f}")

    fig.suptitle("Cross-recovery matrix (diagonal = self-recovery, "
                 "off-diagonal = leakage)", fontsize=9.5)
    figstyle.finish(fig, op.join(HERE, "notes", "figures", "recovery_cloud.pdf"),
                    trim=False)


if __name__ == "__main__":
    main()
