"""Aggregate the RDM 2x2 grid (w_d free|fixed  x  contaminant on|off) into a
compact summary for plotting. Runs on the cluster (traces are large + remote).

For each available trace: group-level convergence (min ESS, max r-hat, div),
the dyscalc-control contrasts on the two noise SDs (link scale), and -- where
w_d is free -- the joint (w_d, perceptual_noise_sd) intercept draws so we can
draw the ridge that fixing w_d removes.

  python collect_rdm_grid.py --out-npz notes/data/rdm_grid.npz \
      --out-json notes/data/rdm_grid.json
"""
import argparse, json, os, warnings
warnings.filterwarnings("ignore")
import arviz as az, numpy as np

GRID = {                                   # cell -> trace label
    "free_cont":    "rdm_pm_cont_hn",      # w_d free,  contaminant on
    "fixed_cont":   "rdm_nowd_cont_hn",    # w_d fixed, contaminant on
    "free_nocont":  "rdm_pm_nocont_hn",    # w_d free,  contaminant off
    "fixed_nocont": "rdm_nowd_nocont_hn",  # w_d fixed, contaminant off
}
NOISE = ["perceptual_noise_sd", "memory_noise_sd"]


def mu_draws(post, param, level):
    da = post[f"{param}_mu"]
    rdim = [d for d in da.dims if d.endswith("_regressors")][0]
    return da.sel({rdim: level}).values.ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", default="/shares/zne.uzh/gdehol/dyscalculic_ddm")
    p.add_argument("--out-npz", required=True)
    p.add_argument("--out-json", required=True)
    a = p.parse_args()

    summary, npz = {}, {}
    for cell, lab in GRID.items():
        path = os.path.join(a.trace_dir, f"model-{lab}_trace.nc")
        if not os.path.exists(path):
            summary[cell] = {"label": lab, "status": "missing"}
            print(cell, lab, "MISSING"); continue
        idata = az.from_netcdf(path)
        post = idata.posterior
        mu = [v for v in post.data_vars if v.endswith("_mu")]
        s = az.summary(idata, var_names=mu)
        div = int(idata.sample_stats["diverging"].values.sum()) \
            if "diverging" in idata.sample_stats else -1
        rec = {"label": lab, "status": "ok",
               "min_ess": int(s["ess_bulk"].min()),
               "max_rhat": round(float(s["r_hat"].max()), 3), "div": div,
               "w_d_free": any(v.startswith("w_d") for v in post.data_vars)}
        # noise contrasts (dyscalc - control, link scale)
        lvl = None
        for v in mu:
            rdim = [d for d in post[v].dims if d.endswith("_regressors")][0]
            lvl = [c for c in post[v].coords[rdim].values if c != "Intercept"]
            if lvl:
                lvl = lvl[0]; break
        for prm in NOISE:
            b = mu_draws(post, prm, lvl)
            lo, hi = az.hdi(b, hdi_prob=0.94)
            rec[prm] = {"mean": float(b.mean()), "lo": float(lo),
                        "hi": float(hi), "Pgt0": float((b > 0).mean())}
        # ridge: joint intercept draws of (w_d, perceptual_noise_sd)
        if rec["w_d_free"]:
            wd = mu_draws(post, "w_d", "Intercept")
            pn = mu_draws(post, "perceptual_noise_sd", "Intercept")
            r = float(np.corrcoef(wd, pn)[0, 1])
            rec["wd_pn_corr"] = round(r, 3)
            # thin to 2000 pts for a light scatter
            idx = np.linspace(0, len(wd) - 1, min(2000, len(wd))).astype(int)
            npz[f"{cell}__wd"] = wd[idx]; npz[f"{cell}__pn"] = pn[idx]
        summary[cell] = rec
        print(cell, json.dumps({k: rec[k] for k in
              ("min_ess", "max_rhat", "div", "w_d_free")}))

    os.makedirs(os.path.dirname(a.out_npz) or ".", exist_ok=True)
    np.savez(a.out_npz, **npz)
    json.dump(summary, open(a.out_json, "w"), indent=2)
    print("wrote", a.out_npz, a.out_json)


if __name__ == "__main__":
    main()
