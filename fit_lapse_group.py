"""Fit the REAL dyscalculic ddm_pm (group contrast) with the feature/ddm-lapse
recipe: fixed-5% RT contaminant + configurable group-SD prior + init choice.

Adapted from the other agent's examples/dyscalculic_ddm_debug/fit_real_ddm.py,
but with the GROUP covariate (the science), not intercept-only. Imports bauer
from the feature/ddm-lapse worktree (BAUER_LAPSE env or default ~/bauer-lapse).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.environ.get("BAUER_LAPSE", os.path.expanduser("~/bauer-lapse")))
import numpy as np
import pandas as pd
import arviz as az
from bauer.models import (DDMMagnitudeComparisonRegressionModel as Plain,
                          DDMMagnitudeComparisonLapseRegressionModel as Lapse)

p = argparse.ArgumentParser()
p.add_argument("--data", default=os.path.expanduser("~/git/dyscalculic_ddm/data/magjudge_behavior_DNumRisk.csv"))
p.add_argument("--out", required=True)
p.add_argument("--nc-out", default=None)
p.add_argument("--lapse", action="store_true", help="add fixed-5%% RT contaminant")
p.add_argument("--init", default="mapjitter", choices=("mapjitter", "pathfinder", "priorjitter"))
p.add_argument("--group-sd-dist", default="halfnormal", choices=("halfcauchy", "halfnormal"))
p.add_argument("--n-subjects", type=int, default=0, help="0 = all 66")
p.add_argument("--regress-a", action="store_true", help="also regress threshold a on group (ddm_group)")
p.add_argument("--draws", type=int, default=1000)
p.add_argument("--tune", type=int, default=2000)
p.add_argument("--chains", type=int, default=4)
p.add_argument("--target-accept", type=float, default=0.99)
a = p.parse_args()

df = pd.read_csv(a.data)
df["choice"] = df["chose_n2"].astype(bool)
df = df[df["rt"] >= 0.20].set_index(["subject", "run", "trial_nr"])
if a.n_subjects > 0:
    keep = sorted(df.index.get_level_values("subject").unique())[: a.n_subjects]
    df = df.loc[df.index.get_level_values("subject").isin(keep)]
n_subj = df.index.get_level_values("subject").nunique()

reg = {"perceptual_noise_sd": "C(group)", "memory_noise_sd": "C(group)"}
if a.regress_a:
    reg["a"] = "C(group)"
common = dict(paradigm=df.reset_index(), regressors=reg, fit_prior=True,
              fit_separate_evidence_sd=True, memory_model="shared_perceptual_noise",
              fit_v_scale=False, fix_z=True)
model = (Lapse(**common) if a.lapse else Plain(**common))
if a.lapse:
    model.p_outlier = 0.05                 # FIXED contaminant (HSSM default)
    model.free_parameters = model.get_free_parameters()
model.group_sd_dist = a.group_sd_dist
model.build_estimation_model(data=df, hierarchical=True)

print(f"=== ddm_pm group | lapse={a.lapse} | init={a.init} | "
      f"group_sd={a.group_sd_dist} | n_subj={n_subj} ===", flush=True)
t0 = time.time()
idata = model.sample(draws=a.draws, tune=a.tune, chains=a.chains,
                     target_accept=a.target_accept, backend="numpyro",
                     chain_method="vectorized", find_init=a.init)
elapsed = time.time() - t0

mu = ["perceptual_noise_sd_mu", "memory_noise_sd_mu", "a_mu", "t0_mu"]
mu = [v for v in mu if v in idata.posterior]
s = az.summary(idata, var_names=mu, round_to=4)
div = int(idata.sample_stats["diverging"].values.sum()) if "diverging" in idata.sample_stats else -1
res = dict(lapse=a.lapse, init=a.init, group_sd=a.group_sd_dist, n_subj=int(n_subj),
           regress_a=a.regress_a, elapsed_sec=elapsed,
           max_rhat_mu=float(s["r_hat"].max()), min_ess_mu=float(s["ess_bulk"].min()),
           divergences=div)
for pp in ["perceptual_noise_sd", "memory_noise_sd"]:
    d = [x for x in idata.posterior[f"{pp}_mu"].dims if x.endswith("_regressors")][0]
    lv = [c for c in idata.posterior.coords[d].values if c != "Intercept"][0]
    b = idata.posterior[f"{pp}_mu"].sel({d: lv}).values.ravel()
    res[f"{pp}_delta"] = float(b.mean()); res[f"{pp}_Pgt0"] = float((b > 0).mean())
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print(json.dumps(res, indent=2))
if a.nc_out:
    az.to_netcdf(idata, a.nc_out)
    print("saved nc ->", a.nc_out)
