"""Head-to-head: does a FIXED group contrast (random intercept only) give
tighter group-effect estimates than the current random-slope-on-group?

Same noise-deficit RDM (perceptual+memory ~ group), same prior, two
parameterizations:
  --mode random : regressors={σ:'C(group)'}            (random slope on group; current)
  --mode fixed  : fixed_regressors={σ:'C(group)'},     (fixed group contrast,
                  random_regressors={σ:'1'}             random intercept only)

Imports the random-effects-api bauer (BAUER_REFX) which has the fixed/random
split. Reports the group-contrast posterior mean / sd / ESS so we can compare
the WIDTH (sd) of the σ_perc and σ_mem group effects.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.environ["BAUER_REFX"])               # fixed/random API bauer
sys.path.insert(0, os.path.expanduser("~/git/dyscalculic_ddm"))
import numpy as np, pandas as pd, arviz as az
from fit_ddm import RaceMagnitudeComparisonRegressionModel as Race

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["random", "fixed"], required=True)
p.add_argument("--out", required=True)
p.add_argument("--n-subjects", type=int, default=0)
a = p.parse_args()

df = pd.read_csv(os.path.expanduser("~/git/dyscalculic_ddm/data/magjudge_behavior_DNumRisk.csv"))
df["choice"] = df["chose_n2"].astype(bool)
df = df[df["rt"] >= 0.20].set_index(["subject", "run", "trial_nr"])
if a.n_subjects > 0:
    keep = sorted(df.index.get_level_values("subject").unique())[: a.n_subjects]
    df = df.loc[df.index.get_level_values("subject").isin(keep)]

noise = {"perceptual_noise_sd": "C(group)", "memory_noise_sd": "C(group)"}
common = dict(paradigm=df.reset_index(), fit_separate_evidence_sd=True,
              memory_model="shared_perceptual_noise")
if a.mode == "random":
    m = Race(regressors=noise, **common)                   # random slope on group
else:
    m = Race(fixed_regressors=noise,                       # fixed group contrast
             random_regressors={"perceptual_noise_sd": "1", "memory_noise_sd": "1"},
             **common)
m.build_estimation_model(data=df, hierarchical=True)
idata = m.sample(draws=1000, tune=2000, chains=4, target_accept=0.95,
                 backend="numpyro", chain_method="vectorized")

s = az.summary(idata)
res = {"mode": a.mode, "n_subj": int(df.index.get_level_values("subject").nunique())}
for pp in ["perceptual_noise_sd", "memory_noise_sd"]:
    P = idata.posterior[f"{pp}_mu"]
    d = [x for x in P.dims if x.endswith("_regressors")][0]
    lv = [c for c in P.coords[d].values if c != "Intercept"][0]
    b = P.sel({d: lv}).values.ravel()
    res[f"{pp}_mean"] = float(b.mean())
    res[f"{pp}_sd"] = float(b.std())          # <-- the WIDTH we care about
    res[f"{pp}_hdi94w"] = float(np.diff(az.hdi(b, hdi_prob=0.94))[0])
    res[f"{pp}_ess"] = float(az.ess(P.sel({d: lv})).to_array().values.ravel()[0])
mu = s[s.index.str.contains("_mu")]
res["max_rhat_mu"] = float(mu.r_hat.max())
res["div"] = int(idata.sample_stats["diverging"].values.sum())
json.dump(res, open(a.out, "w"), indent=2)
print(json.dumps(res, indent=2))
