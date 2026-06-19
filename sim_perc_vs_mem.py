"""Forward simulations: how does predicted behaviour change when you raise
perceptual-only vs memory-only encoding noise — for each model class?

The point: identifiability is decided by whether the two noises leave DISTINCT
behavioural fingerprints. We sweep a baseline + a perceptual-only deficit + a
memory-only deficit, **matched to the same overall accuracy** so any difference
is shape, not level, and read out:

  (a) CHOICE model — compression curve P(chose n2) vs n1
  (b) DDM          — compression curve P(chose n2) vs n1
  (c) DDM (standard, memory->drift) — RT quantiles, correct vs error
  (d) sv-DDM (memory->across-trial sv) — RT quantiles, correct vs error

Top row = choice-level signatures (do +perc and +mem look different?).
Bottom row = RT-distribution signatures (does sv break the tie?).

Output: notes/figures/sim_perc_vs_mem.pdf
"""
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bauer.models import MagnitudeComparisonModel, DDMMagnitudeComparisonModel
import figstyle

figstyle.apply()
HERE = op.dirname(op.abspath(__file__))

C_BASE = "#7F7F7F"      # baseline grey
C_PERC = "#3B8C8C"      # perceptual = teal
C_MEM = "#D98A29"       # memory = orange
QS = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
A_DDM, T0_DDM = 1.5, 0.35
TARGET_ACC = 0.80       # matched deficit accuracy (baseline ~0.88)


def grid_paradigm(n_rep=60, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([5, 7, 10, 14, 20, 28])
    ratios = np.exp(np.linspace(np.log(0.45), np.log(2.2), 9))
    rows = []
    t = 0
    for n1 in base:
        for r in ratios:
            n2 = max(2, round(n1 * r))
            for _ in range(n_rep):
                rows.append((1, t, n1, n2)); t += 1
    df = pd.DataFrame(rows, columns=["subject", "trial_nr", "n1", "n2"])
    return df.set_index(["subject", "trial_nr"])


def make_model(kind):
    """kind in {'choice','ddm','svddm'} -> (model, is_ddm)."""
    if kind == "choice":
        return MagnitudeComparisonModel(
            None, fit_separate_evidence_sd=True,
            memory_model="shared_perceptual_noise"), False
    sv = kind == "svddm"
    return DDMMagnitudeComparisonModel(
        None, fit_separate_evidence_sd=True,
        memory_model="shared_perceptual_noise", memory_as_sv=sv), True


def par_df(perc, mem, is_ddm):
    d = {"perceptual_noise_sd": perc, "memory_noise_sd": mem}
    if is_ddm:
        d["a"] = A_DDM; d["t0"] = T0_DDM
    return pd.DataFrame(d, index=pd.Index([1], name="subject"))


def simulate(model, paradigm, perc, mem, is_ddm, seed, n_samples=20):
    par = par_df(perc, mem, is_ddm)
    if is_ddm:
        sim = model.simulate(paradigm, par, n_samples=n_samples, random_seed=seed)
        sim = sim.reset_index()
        sim["choice"] = sim["simulated_choice"].astype(bool)
        sim["rt"] = sim["simulated_rt"].astype(float)
    else:
        sim = model.simulate(paradigm, par, n_samples=n_samples).reset_index()
        sim["choice"] = sim["simulated_choice"].astype(bool)
        sim["rt"] = np.nan
    sim["correct"] = sim["choice"] == (sim["n2"] > sim["n1"])
    return sim


def match_accuracy(model, paradigm, is_ddm, vary, fixed_val, target, seed=1):
    """Bisection on the `vary` noise ('perc'|'mem') to hit target accuracy."""
    lo, hi = 0.03, 0.9
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        perc, mem = (mid, fixed_val) if vary == "perc" else (fixed_val, mid)
        acc = simulate(model, paradigm, perc, mem, is_ddm, seed, n_samples=6)["correct"].mean()
        if acc > target:        # too easy -> need more noise
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def compression_curve(sim):
    """P(chose n2) as a function of n1 (collapsing ratio) — the compression."""
    return sim.groupby("n1")["choice"].mean()


def rt_quantiles(sim):
    out = {}
    for lab, m in [("correct", sim["correct"]), ("error", ~sim["correct"])]:
        rt = sim.loc[m, "rt"].dropna()
        out[lab] = np.quantile(rt, QS) if len(rt) > 20 else np.full(len(QS), np.nan)
    return out


def main():
    paradigm = grid_paradigm()
    # Baseline encoding (control-like): small perceptual + modest memory.
    base_perc, base_mem = 0.08, 0.13

    # --- calibrate matched-accuracy deficits per model class ---
    cal = {}
    for kind in ["choice", "ddm", "svddm"]:
        model, is_ddm = make_model(kind)
        perc_def = match_accuracy(model, paradigm, is_ddm, "perc", base_mem, TARGET_ACC)
        mem_def = match_accuracy(model, paradigm, is_ddm, "mem", base_perc, TARGET_ACC)
        cal[kind] = (model, is_ddm, perc_def, mem_def)
        print(f"{kind:6s}: +perc deficit perc={perc_def:.3f}  "
              f"+mem deficit mem={mem_def:.3f}")

    # --- simulate the three conditions for each model ---
    def three(kind):
        model, is_ddm, pd_, md_ = cal[kind]
        return {
            "base": simulate(model, paradigm, base_perc, base_mem, is_ddm, 11),
            "perc": simulate(model, paradigm, pd_, base_mem, is_ddm, 12),
            "mem":  simulate(model, paradigm, base_perc, md_, is_ddm, 13),
        }
    sims = {k: three(k) for k in ["choice", "ddm", "svddm"]}

    # --- figure ---
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0), constrained_layout=True)
    conds = [("base", C_BASE, "Baseline"), ("perc", C_PERC, "+ Perceptual"),
             ("mem", C_MEM, "+ Memory")]

    def compression_panel(ax, kind, title, letter, label=False):
        for c, col, lab in conds:
            cc = compression_curve(sims[kind][c])
            ax.plot(cc.index, cc.values, "-o", color=col, label=lab, ms=3)
            if label:
                ax.text(cc.index[-1] + 0.8, cc.values[-1], lab, color=col,
                        fontsize=7, va="center")
        ax.set_xticks([5, 10, 20, 28])
        ax.set_xlabel("First number n1"); ax.set_ylabel("P(chose n2)")
        ax.set_title(title)
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        figstyle.panel_letter(ax, letter)

    # TOP ROW — compression (choice proportions) for all three model classes
    compression_panel(axes[0, 0], "choice", "Choice model", "a")
    compression_panel(axes[0, 1], "ddm", "DDM (memory→drift)", "b")
    compression_panel(axes[0, 2], "svddm", "sv-DDM (memory→sv)", "c", label=True)

    # BOTTOM ROW — RT quantiles correct vs error, +perc vs +mem
    def rt_panel(ax, kind, title, letter, label=False):
        for c, col, lab in [("perc", C_PERC, "+ Perceptual"), ("mem", C_MEM, "+ Memory")]:
            q = rt_quantiles(sims[kind][c])
            ax.plot(QS, q["correct"], "-o", color=col, ms=3)
            ax.plot(QS, q["error"], "--s", color=col, ms=3, mfc="white")
            if label:
                ax.text(QS[-1] + 0.02, q["error"][-1], lab, color=col, fontsize=7,
                        va="center")
        ax.set_xlabel("RT quantile"); ax.set_ylabel("RT (s)")
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        ax.set_title(title)
        figstyle.panel_letter(ax, letter)

    rt_panel(axes[1, 0], "ddm", "DDM — RT", "d")
    rt_panel(axes[1, 1], "svddm", "sv-DDM — RT", "e", label=True)
    axes[1, 0].text(0.04, 0.96, "Solid = correct\nDashed = error",
                    transform=axes[1, 0].transAxes, fontsize=7, va="top")

    # (f) error−correct median-RT gap: the slow-error magnitude, two variants
    ax = axes[1, 2]
    x = np.arange(2)
    for i, (c, col, lab) in enumerate([("perc", C_PERC, "+ Perc"), ("mem", C_MEM, "+ Mem")]):
        gaps = []
        for kind in ["ddm", "svddm"]:
            q = rt_quantiles(sims[kind][c])
            gaps.append(q["error"][2] - q["correct"][2])   # q50 error − correct
        ax.plot(x + (i - 0.5) * 0.12, gaps, "o-", color=col, ms=5, label=lab)
        ax.text(x[-1] + 0.08, gaps[-1], lab, color=col, fontsize=7, va="center")
    ax.set_xticks(x); ax.set_xticklabels(["DDM", "sv-DDM"])
    ax.set_ylabel("Error − correct\nmedian RT (s)")
    ax.set_title("Slow-error magnitude")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    figstyle.panel_letter(ax, "f")

    fig.suptitle("Perceptual vs memory deficit (matched accuracy): distinct "
                 "behavioural fingerprints", fontsize=11)
    figstyle.finish(fig, op.join(HERE, "notes", "figures", "sim_perc_vs_mem.pdf"))


if __name__ == "__main__":
    main()
