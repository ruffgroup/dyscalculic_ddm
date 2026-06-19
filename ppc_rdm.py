"""Posterior-predictive cache for the regression RDM, working around two
bauer simulator gaps:

  1. model.ppc hands the regression-RDM a mixed (66x2 / 66x1) parameter dict
     that its simulate() can't DataFrame-ify  -> we instead collapse the
     regression posterior to REALIZED per-subject parameters and simulate from
     a plain (non-regression) RaceDiffusionMagnitudeComparisonModel.
  2. the Wald sampler fails (mean = a/v <= 0) when the advantage drift goes
     non-positive on some trials  -> we monkeypatch _sample_wald_race_2 to
     floor the drift at a small positive value (the positivity the likelihood
     already assumes). Simulation-only; the library is not modified on disk.

Writes results/ppc_<model>.parquet in the same format ppc_choice.py / qpp.py
expect, so the n1-PPC and SSM-style QPP just consume it.

    ~/mambaforge/envs/bauer/bin/python ppc_rdm.py --n-draws 150
"""
import argparse

import arviz as az
import numpy as np
import pandas as pd

import bauer.models.race as race
from bauer.models import RaceDiffusionMagnitudeComparisonModel

# --- monkeypatch: floor per-accumulator drift before the Wald draw ----------
_orig_sampler = race._sample_wald_race_2
def _floor(x, eps=1e-3):
    return np.clip(np.nan_to_num(np.asarray(x, float), nan=eps, posinf=1e6,
                                 neginf=eps), eps, None)
def _safe_sampler(v1, v2, sigma1, sigma2, a, t0, n_samples=1, rng=None):
    # Wald mean = a/v, shape = a^2/sigma^2: all need to be finite & positive.
    return _orig_sampler(_floor(v1), _floor(v2), _floor(sigma1), _floor(sigma2),
                         _floor(a), np.nan_to_num(np.asarray(t0, float)),
                         n_samples=n_samples, rng=rng)
race._sample_wald_race_2 = _safe_sampler


def main(nc, n_draws, out):
    df = pd.read_csv("data/magjudge_behavior_DNumRisk.csv")
    df = df[df["rt"] >= 0.20].copy()
    df["choice"] = df["choice"].astype(bool)
    df["chose_n2"] = df["chose_n2"].astype(bool)
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    df = df.set_index(["subject", "run", "trial_nr"])

    plain = RaceDiffusionMagnitudeComparisonModel(
        df, fit_prior=False, fit_separate_evidence_sd=True,
        memory_model="shared_perceptual_noise", advantage=True)
    fp = list(plain.free_parameters.keys())

    post = az.from_netcdf(nc).posterior
    subs = post.coords["subject"].values
    dys = (df.reset_index().groupby("subject")["group"].first().loc[subs]
           == "dyscalc").values.astype(float)

    def realized(param, ci, di):
        da = post[param].isel(chain=ci, draw=di)
        rd = [d for d in da.dims if d.endswith("_regressors")]
        if not rd:
            return da.values
        icpt = da.sel({rd[0]: "Intercept"}).values
        gl = [l for l in post.coords[rd[0]].values if l != "Intercept"]
        return icpt + (dys * da.sel({rd[0]: gl[0]}).values if gl else 0.0)

    nchain, ndraw = post.sizes["chain"], post.sizes["draw"]
    rng = np.random.default_rng(0)
    flat = rng.choice(nchain * ndraw, min(n_draws, nchain * ndraw), replace=False)

    sims = []
    for k, f in enumerate(flat):
        ci, di = int(f // ndraw), int(f % ndraw)
        pars = pd.DataFrame({p: realized(p, ci, di) for p in fp},
                            index=pd.Index(subs, name="subject"))
        sim = plain.simulate(df, pars, n_samples=1, random_seed=int(f))
        sim = sim.reset_index()
        sim["ppc_sample"] = k
        sims.append(sim)
        if (k + 1) % 25 == 0:
            print(f"  simulated draw {k + 1}/{len(flat)}")
    ppc = pd.concat(sims, ignore_index=True)
    ppc.to_parquet(out)
    print(f"Wrote {out}: {len(ppc)} rows, cols={list(ppc.columns)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nc", default="results/model-rdm_group_trace.nc")
    p.add_argument("--n-draws", type=int, default=150)
    p.add_argument("--out", default="results/ppc_rdm_group.parquet")
    args = p.parse_args()
    main(args.nc, args.n_draws, args.out)
