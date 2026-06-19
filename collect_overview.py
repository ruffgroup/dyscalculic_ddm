"""Aggregate the cross-model verdict (choice / DDM / RDM) into a compact summary
for the overview PDF. Runs on the cluster (big remote traces + data + bauer).

Per model: group contrasts (dyscalc - control) on perceptual/memory noise,
threshold a and non-decision t0; LOO (within-family); and PPC summaries --
  * P(chose n2) vs n1            (compression, all models)
  * median RT vs difficulty      (chronometric, RT models)
  * median RT vs magnitude n1+n2 (stake-size effect, RT models)
Writes a small json + npz so the figure is rendered locally in house style.

  JAX_PLATFORMS=cpu PYTHONPATH=/home/gdehol/bauer-030 python collect_overview.py \
      --out-npz notes/data/overview.npz --out-json notes/data/overview.json
"""
import argparse, json, os
import os.path as op
import warnings
warnings.filterwarnings("ignore")
import arviz as az, numpy as np, pandas as pd
import fit_ddm
from bauer.utils.math import softplus_np, logistic_np


def _to_real(x, tname):
    """Map a link-scale draw to natural units via the parameter's transform."""
    if tname and "softplus" in tname:
        return softplus_np(x)
    if tname == "logistic":
        return logistic_np(x)
    return x

D = "/shares/zne.uzh/gdehol/dyscalculic_ddm"
# tag -> (build spec, trace label, family, pretty)
# tag -> (build spec, trace label, family, pretty). Two noise decompositions:
# shared perceptual/memory, and independent n1/n2 (the *_n1n2 tags).
MODELS = [
    ("choice", "choice:perceptual_noise_sd,memory_noise_sd", "choice_pm_hn",
     "choice", "Choice"),
    ("ddm", "ddm:perceptual_noise_sd,memory_noise_sd,a,t0", "ddm_full_cont_hn",
     "rt", "DDM"),
    ("rdm", "rdm_nowd:perceptual_noise_sd,memory_noise_sd,a,t0", "rdm_full_cont_hn",
     "rt", "RDM"),
    ("choice_n1n2", "choice:n1_evidence_sd,n2_evidence_sd", "choice_n1n2_hn",
     "choice", "Choice"),
    ("ddm_n1n2", "ddm:n1_evidence_sd,n2_evidence_sd,a,t0", "ddm_n1n2_full_cont_hn",
     "rt", "DDM"),
    ("rdm_n1n2", "rdm_nowd:n1_evidence_sd,n2_evidence_sd,a,t0", "rdm_n1n2_full_cont_hn",
     "rt", "RDM"),
    ("rdm_max", "rdm_nowd:perceptual_noise_sd,memory_noise_sd,a,t0,w_0,w_s",
     "rdm_max_cont_hn", "rt", "RDM (full drift)"),
    ("rdm_n1n2_max", "rdm_nowd:n1_evidence_sd,n2_evidence_sd,a,t0,w_0,w_s",
     "rdm_n1n2_max_cont_hn", "rt", "RDM (full drift)"),
]
PARAMS = ["perceptual_noise_sd", "memory_noise_sd",
          "n1_evidence_sd", "n2_evidence_sd", "a", "t0", "w_0", "w_s"]
N_PPC = 60
NBIN = 6


def load_df():
    df = pd.read_csv(op.expanduser("~/git/dyscalculic_ddm/data/magjudge_behavior_DNumRisk.csv"))
    df = df[df["rt"] >= 0.20].copy()
    df["choice"] = df["chose_n2"].astype(bool)
    if df["group"].dtype != object:
        df["group"] = df["group"].map({0: "control", 1: "dyscalc"})
    df["logratio"] = np.log(df["n2"]) - np.log(df["n1"])   # natural decision var
    df["difficulty"] = df["logratio"].abs()
    df["magnitude"] = df["n1"] + df["n2"]
    return df.set_index(["subject", "run", "trial_nr"])


def bin_edges(series, nbin=NBIN):
    return np.unique(np.quantile(series, np.linspace(0, 1, nbin + 1)))


def contrasts(idata):
    out = {}
    post = idata.posterior
    for prm in PARAMS:
        if f"{prm}_mu" not in post:
            continue
        da = post[f"{prm}_mu"]
        rdim = [d for d in da.dims if d.endswith("_regressors")]
        if not rdim:
            continue
        levels = [str(c) for c in da.coords[rdim[0]].values]
        ctrl = [c for c in levels if "control" in c]
        dys = [c for c in levels if "dyscalc" in c]
        if ctrl and dys:                       # cell-means: dyscalc - control
            b = (da.sel({rdim[0]: dys[0]}).values
                 - da.sel({rdim[0]: ctrl[0]}).values).ravel()
        else:                                  # legacy: single contrast coefficient
            lv = [c for c in levels if c != "Intercept"]
            if not lv:
                continue
            b = da.sel({rdim[0]: lv[0]}).values.ravel()
        lo, hi = az.hdi(b, hdi_prob=0.94)
        out[prm] = {"mean": float(b.mean()), "lo": float(lo), "hi": float(hi),
                    "Pgt0": float((b > 0).mean())}
    return out


def rt_vs(sim, df, col, edges, npz, tag):
    """Median simulated RT per (group, bin of `col`): PPC band + observed."""
    s = sim.join(df[[col, "group"]])
    s["bin"] = pd.cut(s[col], edges, include_lowest=True)
    per = (s.groupby(["ppc_sample", "group", "bin"], observed=True)["simulated_rt"]
             .median().reset_index())
    centers = df.groupby(pd.cut(df[col], edges, include_lowest=True),
                         observed=True)[col].mean()
    for (g, b), ss in per.groupby(["group", "bin"], observed=True):
        v = ss["simulated_rt"].values
        if not len(v):
            continue
        lo, hi = az.hdi(v, hdi_prob=0.94)
        npz.setdefault(f"{tag}__{col}__{g}__x", []).append(float(centers.get(b, np.nan)))
        npz.setdefault(f"{tag}__{col}__{g}__med", []).append(float(np.median(v)))
        npz.setdefault(f"{tag}__{col}__{g}__lo", []).append(float(lo))
        npz.setdefault(f"{tag}__{col}__{g}__hi", []).append(float(hi))


def group_sds(idata):
    """Per-group between-subject SDs (tau_control, tau_dyscalc) for each
    cell-means parameter, plus the tau_dyscalc - tau_control contrast. Tests
    whether dyscalcs are also more HETEROGENEOUS, not just higher on average."""
    out = {}
    post = idata.posterior
    for prm in PARAMS:
        sdname = f"{prm}_sd"
        if sdname not in post:
            continue
        da = post[sdname]
        rdim = [d for d in da.dims if d.endswith("_regressors") or d.endswith("_re")]
        if not rdim:
            continue
        levels = [str(c) for c in da.coords[rdim[0]].values]
        ctrl = [c for c in levels if "control" in c]
        dys = [c for c in levels if "dyscalc" in c]
        if not (ctrl and dys):                 # only cell-means has per-group SDs
            continue
        rec = {}
        for g, lv in (("control", ctrl[0]), ("dyscalc", dys[0])):
            b = da.sel({rdim[0]: lv}).values.ravel()
            lo, hi = az.hdi(b, hdi_prob=0.94)
            rec[g] = {"mean": float(b.mean()), "lo": float(lo), "hi": float(hi)}
        bc = (da.sel({rdim[0]: dys[0]}).values
              - da.sel({rdim[0]: ctrl[0]}).values).ravel()
        lo, hi = az.hdi(bc, hdi_prob=0.94)
        rec["contrast"] = {"mean": float(bc.mean()), "lo": float(lo),
                           "hi": float(hi), "Pgt0": float((bc > 0).mean())}
        out[prm] = rec
    return out


def raw_values(idata, model, params):
    """Per-group population value in NATURAL units (transform of the cell-means
    mu), so you can read effect sizes directly (e.g. sigma_n1 in real units)."""
    out = {}
    fp = getattr(model, "free_parameters", {}) or {}
    post = idata.posterior
    for prm in params:
        if f"{prm}_mu" not in post:
            continue
        da = post[f"{prm}_mu"]
        rdim = [d for d in da.dims if d.endswith("_regressors")]
        if not rdim:
            continue
        levels = [str(c) for c in da.coords[rdim[0]].values]
        ctrl = [c for c in levels if "control" in c]
        dys = [c for c in levels if "dyscalc" in c]
        if not (ctrl and dys):
            continue
        tname = fp.get(prm, {}).get("transform", "identity")
        rec = {"transform": tname}
        for g, lv in (("control", ctrl[0]), ("dyscalc", dys[0])):
            b = _to_real(da.sel({rdim[0]: lv}).values.ravel(), tname)
            lo, hi = az.hdi(b, hdi_prob=0.94)
            rec[g] = {"mean": float(b.mean()), "lo": float(lo), "hi": float(hi)}
        out[prm] = rec
    return out


def _stat(v):
    lo, hi = az.hdi(v, hdi_prob=0.94)
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def noise_asymmetry(idata, sigma_p):
    """The n1/n2 asymmetry that drives compression, three ways, per group:
      - diff      sigma_n1 - sigma_n2          (SD difference, scale-dependent)
      - logratio  log(sigma_n1 / sigma_n2)     (scale-invariant)
      - w1/w2/gap reliability weights w = sigma_p^2/(sigma_p^2+sigma_e^2) and the
        gap w2 - w1 -- the CORRECT (variance-based, prior-aware, saturating)
        driver of differential compression.
    Group contrasts (dyscalc - control) for diff / logratio / gap are per-draw."""
    post = idata.posterior
    if "n1_evidence_sd_mu" not in post or "n2_evidence_sd_mu" not in post:
        return None
    da1, da2 = post["n1_evidence_sd_mu"], post["n2_evidence_sd_mu"]
    rd1 = [d for d in da1.dims if d.endswith("_regressors")][0]
    rd2 = [d for d in da2.dims if d.endswith("_regressors")][0]
    lv = [str(c) for c in da1.coords[rd1].values]
    glev = {}
    for g in ("control", "dyscalc"):
        m = [c for c in lv if g in c]
        if not m:
            return None
        glev[g] = m[0]
    sp2 = float(sigma_p) ** 2
    keys = ("diff", "logratio", "w1", "w2", "gap")
    out = {k: {} for k in keys}
    out["sigma_p"] = float(sigma_p)
    draws = {k: {} for k in keys}
    for g in ("control", "dyscalc"):
        s1 = softplus_np(da1.sel({rd1: glev[g]}).values.ravel())
        s2 = softplus_np(da2.sel({rd2: glev[g]}).values.ravel())
        w1 = sp2 / (sp2 + s1 ** 2)
        w2 = sp2 / (sp2 + s2 ** 2)
        vals = {"diff": s1 - s2, "logratio": np.log(s1) - np.log(s2),
                "w1": w1, "w2": w2, "gap": w2 - w1}
        for k, v in vals.items():
            draws[k][g] = v
            out[k][g] = _stat(v)
    for k in ("diff", "logratio", "gap"):
        c = draws[k]["dyscalc"] - draws[k]["control"]
        out[k]["contrast"] = {**_stat(c), "Pgt0": float((c > 0).mean())}
    return out


def psychometric(sim, df, edges, npz, tag):
    """P(chose n2) vs log(n2/n1), faceted by reference magnitude n1: PPC band."""
    s = sim.join(df[["n1", "logratio", "group"]])
    s["lrbin"] = pd.cut(s["logratio"], edges, include_lowest=True)
    per = (s.groupby(["ppc_sample", "n1", "group", "lrbin"], observed=True)
             ["simulated_choice"].mean().reset_index())
    centers = df.groupby(pd.cut(df["logratio"], edges, include_lowest=True),
                         observed=True)["logratio"].mean()
    for (n1, g, b), ss in per.groupby(["n1", "group", "lrbin"], observed=True):
        v = ss["simulated_choice"].values
        if not len(v):
            continue
        lo, hi = az.hdi(v, hdi_prob=0.94)
        k = f"{tag}__psy__n1{int(n1)}__{g}"
        npz.setdefault(k + "__x", []).append(float(centers.get(b, np.nan)))
        npz.setdefault(k + "__med", []).append(float(np.median(v)))
        npz.setdefault(k + "__lo", []).append(float(lo))
        npz.setdefault(k + "__hi", []).append(float(hi))


def rt_psychometric(sim, df, edges, npz, tag):
    """Median RT vs log(n2/n1), faceted by reference magnitude n1: PPC band."""
    s = sim.join(df[["n1", "logratio", "group"]])
    s["lrbin"] = pd.cut(s["logratio"], edges, include_lowest=True)
    per = (s.groupby(["ppc_sample", "n1", "group", "lrbin"], observed=True)
             ["simulated_rt"].median().reset_index())
    centers = df.groupby(pd.cut(df["logratio"], edges, include_lowest=True),
                         observed=True)["logratio"].mean()
    for (n1, g, b), ss in per.groupby(["n1", "group", "lrbin"], observed=True):
        v = ss["simulated_rt"].values
        if not len(v):
            continue
        lo, hi = az.hdi(v, hdi_prob=0.94)
        k = f"{tag}__rtpsy__n1{int(n1)}__{g}"
        npz.setdefault(k + "__x", []).append(float(centers.get(b, np.nan)))
        npz.setdefault(k + "__med", []).append(float(np.median(v)))
        npz.setdefault(k + "__lo", []).append(float(lo))
        npz.setdefault(k + "__hi", []).append(float(hi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-npz", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--n-ppc", type=int, default=N_PPC)
    a = p.parse_args()
    df = load_df()
    diff_edges, mag_edges = bin_edges(df["difficulty"]), bin_edges(df["magnitude"])
    lr_edges = bin_edges(df["logratio"], 8)
    # shared empirical log-magnitude prior SD, exactly as the model sets it
    sigma_p = (np.log(df["n1"]).std() + np.log(df["n2"]).std()) / 2.0

    summary, npz = {}, {}
    # observed reference curves
    obs_choice = (df.reset_index().groupby(["group", "n1"])["choice"].mean()
                  .rename("p").reset_index())
    for g, s in obs_choice.groupby("group"):
        npz[f"obs__{g}__n1"] = s["n1"].values
        npz[f"obs__{g}__p"] = s["p"].values
    for col, edges in [("difficulty", diff_edges), ("magnitude", mag_edges)]:
        dr = df.reset_index().copy()
        dr["bin"] = pd.cut(dr[col], edges, include_lowest=True)
        centers = dr.groupby("bin", observed=True)[col].mean()
        od = dr.groupby(["group", "bin"], observed=True)["rt"].median().reset_index()
        for g, ss in od.groupby("group"):
            npz[f"obsrt__{col}__{g}__x"] = np.array([centers.get(b, np.nan) for b in ss["bin"]])
            npz[f"obsrt__{col}__{g}__y"] = ss["rt"].values
    # observed psychometric: P(chose n2) vs log(n2/n1), per (n1, group)
    dr = df.reset_index().copy()
    dr["lrbin"] = pd.cut(dr["logratio"], lr_edges, include_lowest=True)
    lcenters = dr.groupby("lrbin", observed=True)["logratio"].mean()
    opsy = dr.groupby(["n1", "group", "lrbin"], observed=True)["choice"].mean().reset_index()
    for (n1, g), ss in opsy.groupby(["n1", "group"]):
        k = f"obs__psy__n1{int(n1)}__{g}"
        npz[k + "__x"] = np.array([lcenters.get(b, np.nan) for b in ss["lrbin"]])
        npz[k + "__p"] = ss["choice"].values
    # observed RT vs log(n2/n1), per (n1, group)
    ortpsy = dr.groupby(["n1", "group", "lrbin"], observed=True)["rt"].median().reset_index()
    for (n1, g), ss in ortpsy.groupby(["n1", "group"]):
        k = f"obs__rtpsy__n1{int(n1)}__{g}"
        npz[k + "__x"] = np.array([lcenters.get(b, np.nan) for b in ss["lrbin"]])
        npz[k + "__y"] = ss["rt"].values

    for tag, spec, lab, fam, pretty in MODELS:
        path = op.join(D, f"model-{lab}_trace.nc")
        if not op.exists(path):
            summary[tag] = {"status": "missing", "label": lab}
            print(tag, "MISSING"); continue
        idata = az.from_netcdf(path)
        rec = {"status": "ok", "pretty": pretty, "family": fam,
               "contrasts": contrasts(idata), "group_sds": group_sds(idata)}
        asym = noise_asymmetry(idata, sigma_p)
        if asym:
            rec["asymmetry"] = asym
        mu = [v for v in idata.posterior.data_vars if v.endswith("_mu")]
        s = az.summary(idata, var_names=mu)
        rec["min_ess"] = int(s["ess_bulk"].min())
        rec["max_rhat"] = round(float(s["r_hat"].max()), 3)
        if "log_likelihood" in idata.groups():
            try:
                lo = az.loo(idata, pointwise=True)
                rec.update(elpd_loo=round(float(lo.elpd_loo), 1),
                           elpd_se=round(float(lo.se), 1),
                           bad_k=int((lo.pareto_k.values > 0.7).sum()),
                           n_obs=int(lo.pareto_k.size))
            except Exception as e:
                rec["loo_err"] = str(e)[:60]
        try:
            m = fit_ddm.build_model(spec, df)
            if "_hn" in lab:
                m.group_sd_dist = "halfnormal"
            m.build_estimation_model(df, hierarchical=True)
            rec["raw"] = raw_values(idata, m, PARAMS)
            sim = m.ppc(df, idata, n_posterior_samples=a.n_ppc, progressbar=False)
            simc = sim.join(df[["n1", "group"]])
            per = (simc.groupby(["ppc_sample", "group", "n1"])["simulated_choice"]
                       .mean().reset_index())
            for (g, n1), ss in per.groupby(["group", "n1"]):
                v = ss["simulated_choice"].values
                lo_, hi_ = az.hdi(v, hdi_prob=0.94)
                npz.setdefault(f"{tag}__{g}__n1", []).append(n1)
                npz.setdefault(f"{tag}__{g}__med", []).append(float(np.median(v)))
                npz.setdefault(f"{tag}__{g}__lo", []).append(float(lo_))
                npz.setdefault(f"{tag}__{g}__hi", []).append(float(hi_))
            psychometric(sim, df, lr_edges, npz, tag)
            if fam == "rt" and "simulated_rt" in sim.columns:
                rt_vs(sim, df, "difficulty", diff_edges, npz, tag)
                rt_vs(sim, df, "magnitude", mag_edges, npz, tag)
                rt_psychometric(sim, df, lr_edges, npz, tag)
            rec["ppc"] = "ok"
        except Exception as e:
            rec["ppc_err"] = str(e)[:90]
            print(tag, "PPC FAILED:", e)
        summary[tag] = rec
        print(tag, json.dumps({k: rec.get(k) for k in
              ("min_ess", "max_rhat", "elpd_loo", "bad_k", "ppc")}))

    npz = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in npz.items()}
    os.makedirs(op.dirname(a.out_npz) or ".", exist_ok=True)
    np.savez(a.out_npz, **npz)
    json.dump(summary, open(a.out_json, "w"), indent=2)
    print("wrote", a.out_npz, a.out_json)


if __name__ == "__main__":
    main()
