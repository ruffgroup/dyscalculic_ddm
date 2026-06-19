"""The RDM 2x2 PDF: does fixing w_d give a clean, well-mixed race model, what do
its noise contrasts look like, and does the contaminant matter for the RDM?

Reads the summary from collect_rdm_grid.py and renders one figure:
  A  2x2 convergence map (min ESS / r-hat) over w_d {free,fixed} x contaminant
  B  the w_d <-> perceptual-noise ridge that fixing w_d removes
  C  noise-deficit contrasts (dyscalc - control) across all four models

  ~/mambaforge/envs/bauer/bin/python plot_rdm_grid.py \
      --npz notes/data/rdm_grid.npz --json notes/data/rdm_grid.json \
      --out notes/figures/rdm_grid.pdf
"""
import argparse, json
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import figstyle

figstyle.apply()
HERE = op.dirname(op.abspath(__file__))

# 2x2 cell layout: rows = w_d, cols = contaminant
ROWS = [("free", "w_d free"), ("fixed", "w_d fixed (≡1)")]
COLS = [("cont", "Contaminant"), ("nocont", "No contaminant")]
PARAMS = [("perceptual_noise_sd", "Perceptual noise"),
          ("memory_noise_sd", "Memory noise")]
ESS_OK = 400      # "usable" threshold


def cell_key(r, c):
    return f"{r}_{c}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="notes/data/rdm_grid.npz")
    p.add_argument("--json", default="notes/data/rdm_grid.json")
    p.add_argument("--out", default="notes/figures/rdm_grid.pdf")
    a = p.parse_args()
    S = json.load(open(op.join(HERE, a.json)))
    Z = np.load(op.join(HERE, a.npz))

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1, 1],
                           hspace=0.55, wspace=0.38,
                           left=0.11, right=0.97, top=0.90, bottom=0.09)

    # ---- A: convergence 2x2 heatmap ----
    axA = fig.add_subplot(gs[0, 0])
    ess = np.full((2, 2), np.nan)
    for i, (rk, _) in enumerate(ROWS):
        for j, (ck, _) in enumerate(COLS):
            rec = S.get(cell_key(rk, ck), {})
            if rec.get("status") == "ok":
                ess[i, j] = rec["min_ess"]
    cmap = LinearSegmentedColormap.from_list("ess", ["#C44E52", "#E8C35A", "#5D8C3F"])
    im = axA.imshow(np.clip(ess, 0, 800), cmap=cmap, vmin=0, vmax=800, aspect="auto")
    for i, (rk, _) in enumerate(ROWS):
        for j, (ck, _) in enumerate(COLS):
            rec = S.get(cell_key(rk, ck), {})
            if rec.get("status") == "ok":
                txt = f"ESS {rec['min_ess']}\nr̂ {rec['max_rhat']:.2f}"
                col = "white" if (np.isnan(ess[i, j]) or ess[i, j] < 250) else "0.1"
                axA.text(j, i, txt, ha="center", va="center", fontsize=9,
                         color=col, fontweight="bold")
            else:
                axA.text(j, i, "—", ha="center", va="center", color="0.5")
    axA.set_xticks([0, 1]); axA.set_xticklabels([c[1] for c in COLS])
    axA.set_yticks([0, 1]); axA.set_yticklabels([r[1] for r in ROWS])
    axA.set_title("Convergence (min ESS, group level)", fontsize=10)
    for s in axA.spines.values():
        s.set_visible(False)
    axA.tick_params(length=0)
    figstyle.panel_letter(axA, "A")

    # ---- B: the ridge ----
    axB = fig.add_subplot(gs[0, 1])
    drew = False
    for rk, ck, col, lab in [("free", "cont", "#8172B2", "w_d free"),
                             ("fixed", "cont", "#5D8C3F", "w_d fixed")]:
        key = cell_key(rk, ck)
        if f"{key}__wd" in Z.files:
            wd, pn = Z[f"{key}__wd"], Z[f"{key}__pn"]
            r = S[key].get("wd_pn_corr")
            axB.scatter(wd, pn, s=4, alpha=0.25, color=col, lw=0, rasterized=True)
            axB.annotate(f"{lab}: r = {r:+.2f}", (0.05, 0.93 if rk == "free" else 0.86),
                         xycoords="axes fraction", color=col, fontsize=8.5,
                         fontweight="bold")
            drew = True
    if not drew:
        axB.text(0.5, 0.5, "w_d-free draws\nnot available", ha="center",
                 va="center", transform=axB.transAxes, color="0.5")
    axB.set_xlabel("w_d (drift gain, group mean)")
    axB.set_ylabel("Perceptual noise SD\n(group mean, link scale)")
    axB.set_title("Why: the w_d ↔ noise ridge", fontsize=10)
    axB.annotate("Fixing w_d≡1\nremoves this axis", (0.95, 0.06),
                 xycoords="axes fraction", ha="right", fontsize=8, color="0.35",
                 style="italic")
    figstyle.panel_letter(axB, "B")

    # ---- C: noise contrasts forest ----
    axC = fig.add_subplot(gs[1, :])
    order = [("fixed", "cont"), ("fixed", "nocont"),
             ("free", "cont"), ("free", "nocont")]
    cellcol = {("fixed", "cont"): "#2C6E49", ("fixed", "nocont"): "#7FB069",
               ("free", "cont"): "#5E4B8B", ("free", "nocont"): "#A99BCB"}
    celllab = {("fixed", "cont"): "w_d fixed · contaminant",
               ("fixed", "nocont"): "w_d fixed · no contaminant",
               ("free", "cont"): "w_d free · contaminant",
               ("free", "nocont"): "w_d free · no contaminant"}
    yticks, ylabels = [], []
    y = 0.0
    for pk, plabel in PARAMS:
        block_top = y
        for (rk, ck) in order:
            rec = S.get(cell_key(rk, ck), {})
            yy = y
            if rec.get("status") == "ok" and pk in rec:
                d = rec[pk]
                col = cellcol[(rk, ck)]
                axC.plot([d["lo"], d["hi"]], [yy, yy], color=col, lw=2.2,
                         solid_capstyle="round", zorder=2)
                axC.scatter([d["mean"]], [yy], s=34, color=col, zorder=3,
                            edgecolor="white", lw=1.0)
                axC.annotate(f"P>0 = {d['Pgt0']:.2f}", (d["hi"], yy),
                             xytext=(6, 0), textcoords="offset points",
                             va="center", fontsize=7.2, color=col)
            yticks.append(yy); ylabels.append(celllab[(rk, ck)])
            y -= 1.0
        # block label
        axC.annotate(plabel, (0.0, (block_top + yy) / 2), xycoords=("axes fraction", "data"),
                     xytext=(-128, 0), textcoords="offset points", fontsize=9.5,
                     fontweight="bold", va="center", rotation=90, annotation_clip=False)
        y -= 0.7
    axC.axvline(0, ls=":", color="0.5", lw=0.9, zorder=1)
    axC.set_yticks(yticks); axC.set_yticklabels(ylabels, fontsize=8)
    axC.set_xlabel("Group contrast  dyscalc − control  (link scale; >0 = more noise in dyscalc)")
    axC.set_title("Noise-deficit contrast is stable across w_d / contaminant choices",
                  fontsize=10)
    axC.set_ylim(y + 0.4, 0.6)
    figstyle.panel_letter(axC, "C")

    fig.suptitle("Race-diffusion model: fixing the drift gain w_d resolves the fit; "
                 "the contaminant is for the DDM, not the RDM",
                 fontsize=10.5, y=0.975)
    figstyle.finish(fig, op.join(HERE, a.out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
