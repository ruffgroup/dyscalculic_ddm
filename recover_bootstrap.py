"""Parametric-bootstrap parameter recovery (posterior-predictive calibration).

For a given model class, one *rep*:
  1. pick one draw k from that model's REAL-data posterior (a "plausible truth",
     including its own fitted group contrast);
  2. simulate one dataset on the REAL paradigm from draw k via the model's own
     faithful `ppc` (no manual parameter reconstruction);
  3. refit the same model class as a group regression;
  4. record TRUE vs RECOVERED group contrasts — the C(group)[dyscalc] slope on
     each noise parameter (model's internal link scale; perc and mem share that
     scale so they are comparable).

Run 100 reps (each a different draw) to get the recovery cloud. One rep per
process so it parallelises as a SLURM array; one row per rep is written to
results/recover_boot/<model>_rep<NN>.csv and aggregated later.

  python recover_bootstrap.py --model choice --rep 0 \
      --trace results/model-choice_group_trace.nc
"""
import argparse
import os
import os.path as op

import arviz as az
import numpy as np
import pandas as pd

import fit_ddm

HERE = op.dirname(op.abspath(__file__))

SPEC = {
    "choice":     dict(refit="choice_group", is_ddm=False, klw=False),
    "choice_klw": dict(refit="choice_group", is_ddm=False, klw=True),
    "ddm":        dict(refit="ddm_group", is_ddm=True, klw=False),
    "ddm_sv":     dict(refit="ddm_sv_group", is_ddm=True, klw=False),
    "rdm":        dict(refit="rdm_group", is_ddm=True, klw=False),
}
CONTRASTS = ["perceptual_noise_sd", "memory_noise_sd"]


def make_model(spec, df):
    """Build the refit model, applying the KLW choice-noise flag if requested.

    The flag must be set BEFORE build_estimation_model so the consistent
    (posterior-mean-SD) normalization is baked into both the generating ppc
    and the refit likelihood."""
    m = fit_ddm.build_model(spec["refit"], df)
    if spec.get("klw"):
        m.consistent_choice_noise = True
    m.build_estimation_model(df, hierarchical=True)
    return m


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SPEC), required=True)
    ap.add_argument("--trace", required=True, help="real-data posterior .nc")
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--backend", default="numpyro")
    ap.add_argument("--target-accept", type=float, default=0.95)
    args = ap.parse_args()
    spec = SPEC[args.model]

    df = load_df()
    model = make_model(spec, df)                          # ppc needs it built
    idata = az.from_netcdf(args.trace)
    post = idata.posterior
    c, d = pick_chain_draw(post.sizes["chain"], post.sizes["draw"], args.rep)

    # ---- TRUE contrast = this draw's group-mu slope (link scale) ----
    true_d = {}
    for p in CONTRASTS:
        rdim = f"{p}_regressors"
        mu = post[f"{p}_mu"]
        true_d[p] = (float(mu.isel(chain=c, draw=d).isel({rdim: 1}).values)
                     if mu.sizes.get(rdim, 1) > 1 else np.nan)

    # ---- simulate one faithful dataset from draw (c, d) via ppc ----
    idata_k = idata.isel(chain=[c], draw=[d])
    sim = model.ppc(df, idata_k, n_posterior_samples=1, progressbar=False)
    sim = sim.reset_index().merge(
        df[["n1", "n2", "group"]].reset_index(),
        on=["subject", "run", "trial_nr"], how="left")
    sim["choice"] = sim["simulated_choice"].astype(bool)
    if spec["is_ddm"]:
        sim["rt"] = sim["simulated_rt"].astype(float)
        sim = sim[sim["rt"] >= 0.20].copy()
    else:
        sim["rt"] = 1.0
    sim_acc = float((sim["choice"] == (sim["n2"] > sim["n1"])).mean())

    fit_df = sim[["subject", "run", "trial_nr", "n1", "n2", "choice", "rt", "group"]]
    fit_df = fit_df.set_index(["subject", "run", "trial_nr"])

    # ---- refit the same model class (same KLW setting) ----
    m = make_model(spec, fit_df)
    ridata = m.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                      target_accept=args.target_accept, backend=args.backend)
    rpost = ridata.posterior

    row = {"model": args.model, "rep": args.rep, "chain": c, "draw": d,
           "sim_acc": sim_acc,
           "max_rhat": float(az.summary(ridata)["r_hat"].max())}
    for p in CONTRASTS:
        rdim = f"{p}_regressors"
        slope = rpost[f"{p}_mu"].isel({rdim: 1}).values.ravel()
        row[f"true_{p}"] = true_d[p]
        row[f"rec_{p}"] = float(slope.mean())
        row[f"recP_{p}"] = float((slope > 0).mean())

    out_dir = op.join(HERE, "results", "recover_boot")
    os.makedirs(out_dir, exist_ok=True)
    out = op.join(out_dir, f"{args.model}_rep{args.rep:03d}.csv")
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"[{args.model}] rep {args.rep} (chain {c} draw {d}): acc={sim_acc:.3f}  "
          f"perc {row['true_perceptual_noise_sd']:+.3f}->{row['rec_perceptual_noise_sd']:+.3f}  "
          f"mem {row['true_memory_noise_sd']:+.3f}->{row['rec_memory_noise_sd']:+.3f}  "
          f"rhat={row['max_rhat']:.2f} -> {out}")


if __name__ == "__main__":
    main()
