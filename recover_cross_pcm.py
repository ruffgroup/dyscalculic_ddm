"""Cross-model recovery: does the pure PCM's paradoxical dyscalc < control
perceptual-noise contrast reflect a genuine perceptual difference, or a
threshold/noise confound that a no-RT choice model can't see through?

Committed reality (notes/data/overview.json): the "choice" tag (pure PCM,
choice:perceptual_noise_sd,memory_noise_sd) finds dyscalc LOWER apparent
noise (ctrl 0.104 vs dys 0.087) -- the reverse of "rdm_max" (the RT model
with a threshold), which finds dyscalc HIGHER noise (ctrl 0.34 vs dys 0.45,
P=1.00) *and* higher threshold (ctrl 0.49 vs dys 0.66, P=1.00). Christian's
hypothesis: dyscalcs compensate for real extra noise with a higher
threshold, and a choice-only model -- which has no threshold parameter to
absorb that -- misreads the resulting (near-)normal accuracy as "less
noise". This script tests that mechanically rather than by correlating
noisy subject-level point estimates (which is underpowered -- see chat).

One rep:
  1. Pick one draw k from the REAL rdm_max posterior (perceptual/memory
     noise AND threshold a AND drift w0/ws, all group-contrasted,
     cell-means) -- the "truth": a world where dyscalc noise really IS
     higher, and dyscalc threshold really IS higher too, both baked into
     that one draw's per-subject parameters.
  2. Simulate choices+RTs on the real paradigm from draw k via rdm_max's
     own `ppc` (no manual parameter reconstruction -- same pattern as
     recover_bootstrap.py).
  3. Drop the RTs -- exactly what fitting the choice-only PCM to real data
     does -- and refit the PCM (same cell-means "choice:perceptual_noise_sd,
     memory_noise_sd" spec collect_overview.py uses for the real "choice"
     tag) to the simulated CHOICES only.
  4. Record the refit PCM's dyscalc-control noise contrast. If the paradox
     is a threshold/noise confound, it should reappear here even though the
     simulation's TRUE noise contrast goes the other way (dys > ctrl).

  python recover_cross_pcm.py --rep 0 \
      --trace /shares/zne.uzh/gdehol/dyscalculic_ddm/model-rdm_max_cont_hn_trace.nc
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd

import fit_ddm

HERE = op.dirname(op.abspath(__file__))
# Exactly the build specs collect_overview.py uses for "rdm_max" (truth) and
# "choice" (refit) -- so results are directly comparable to the committed
# overview.json numbers.
TRUTH_SPEC = "rdm_nowd:perceptual_noise_sd,memory_noise_sd,a,t0,w_0,w_s"
REFIT_SPEC = "choice:perceptual_noise_sd,memory_noise_sd"
CONTRASTS = ["perceptual_noise_sd", "memory_noise_sd", "a", "t0", "w_0", "w_s"]


def load_df():
    df = pd.read_csv(op.join(HERE, "data", "magjudge_behavior_DNumRisk.csv"))
    df = df[df["rt"] >= 0.20].copy()
    df["choice"] = df["chose_n2"].astype(bool)
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    return df.set_index(["subject", "run", "trial_nr"])


def pick_chain_draw(n_chain, n_draw, rep, seed=0):
    """Fixed shuffle -> reproducible, spread-out (chain, draw) for this rep."""
    flat = np.random.default_rng(seed).permutation(n_chain * n_draw)[rep % (n_chain * n_draw)]
    return int(flat // n_draw), int(flat % n_draw)


def group_contrast(mu_da):
    """dyscalc - control on the link scale, by level-NAME match (robust to
    column order) -- same lookup collect_overview.py's contrasts() uses.
    Returns a DataArray retaining whatever dims mu_da still has (e.g.
    chain/draw), so callers can .isel() a single draw or reduce over all."""
    rdim = [dd for dd in mu_da.dims if dd.endswith("_regressors")]
    if not rdim:
        return None
    rdim = rdim[0]
    levels = [str(cc) for cc in mu_da.coords[rdim].values]
    ctrl = [cc for cc in levels if "control" in cc]
    dys = [cc for cc in levels if "dyscalc" in cc]
    if not (ctrl and dys):
        return None
    return mu_da.sel({rdim: dys[0]}) - mu_da.sel({rdim: ctrl[0]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="real rdm_max posterior .nc")
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--backend", default="numpyro")
    ap.add_argument("--target-accept", type=float, default=0.95)
    args = ap.parse_args()

    df = load_df()

    truth_model = fit_ddm.build_model(TRUTH_SPEC, df)
    truth_model.build_estimation_model(df, hierarchical=True)   # ppc needs it built
    idata = az.from_netcdf(args.trace)
    post = idata.posterior
    c, d = pick_chain_draw(post.sizes["chain"], post.sizes["draw"], args.rep)

    # ---- TRUE contrasts = this draw's group-mu slopes (link scale) ----
    true_d = {}
    for p in CONTRASTS:
        key = f"{p}_mu"
        if key not in post:
            true_d[p] = np.nan
            continue
        gc = group_contrast(post[key])
        true_d[p] = float(gc.isel(chain=c, draw=d).values) if gc is not None else np.nan

    # ---- simulate ONE faithful choice+RT dataset from draw (c, d) ----
    idata_k = idata.isel(chain=[c], draw=[d])
    sim = truth_model.ppc(df, idata_k, n_posterior_samples=1, progressbar=False)
    sim = sim.reset_index().merge(
        df[["n1", "n2", "group"]].reset_index(),
        on=["subject", "run", "trial_nr"], how="left")
    sim["choice"] = sim["simulated_choice"].astype(bool)
    sim["rt"] = sim["simulated_rt"].astype(float)
    sim = sim[sim["rt"] >= 0.20].copy()          # same trial-exclusion real data got
    sim_acc = float((sim["choice"] == (sim["n2"] > sim["n1"])).mean())

    fit_df = sim[["subject", "run", "trial_nr", "n1", "n2", "choice", "rt", "group"]]
    fit_df = fit_df.set_index(["subject", "run", "trial_nr"])

    # ---- refit the PCM (choice-only, no threshold) to the simulated CHOICES ----
    refit_model = fit_ddm.build_model(REFIT_SPEC, fit_df)
    refit_model.build_estimation_model(fit_df, hierarchical=True)
    ridata = refit_model.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                                target_accept=args.target_accept, backend=args.backend)
    rpost = ridata.posterior

    row = {"rep": args.rep, "chain": c, "draw": d, "sim_acc": sim_acc,
           "max_rhat": float(az.summary(ridata)["r_hat"].max())}
    for p in CONTRASTS:
        row[f"true_{p}"] = true_d[p]
    for p in ("perceptual_noise_sd", "memory_noise_sd"):
        gc = group_contrast(rpost[f"{p}_mu"])
        vals = gc.values.ravel()
        row[f"rec_{p}"] = float(vals.mean())
        row[f"recP_{p}"] = float((vals > 0).mean())

    out_dir = op.join(HERE, "results", "recover_cross_pcm")
    os.makedirs(out_dir, exist_ok=True)
    out = op.join(out_dir, f"rep{args.rep:03d}.csv")
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"rep {args.rep} (chain {c} draw {d}): sim_acc={sim_acc:.3f}  "
          f"TRUE perc {row['true_perceptual_noise_sd']:+.3f} a {row['true_a']:+.3f}  "
          f"-> REFIT-PCM perc {row['rec_perceptual_noise_sd']:+.3f} "
          f"(P(dys>ctrl)={row['recP_perceptual_noise_sd']:.2f})  "
          f"rhat={row['max_rhat']:.2f} -> {out}")


if __name__ == "__main__":
    main()
