"""Convergence + LOO verdict for a set of fitted traces.

Reads model-<label>_trace.nc files, reports group-level (*_mu) max r-hat /
min ESS / divergences, and (if log_likelihood is present) LOO with a
Pareto-k reliability flag. Run it from the repo on the cluster so it lives
on the shared filesystem (no /tmp node-local gotcha) and is version-tracked:

  srun -c2 --mem 16G --time 20 --account=zne.uzh \
      $HOME/data/conda/envs/bauer_cuda/bin/python summarize_traces.py \
      --trace-dir /shares/zne.uzh/gdehol/dyscalculic_ddm \
      --labels ddm_pm_cont ddm_group_cont rdm_pm_cont rdm_rate_cont rdm_all3_cont \
      --json-out notes/data/contaminant_diag.json

For a LOO *comparison* across a subset (e.g. noise vs rate vs both), pass
--compare with the labels to rank by elpd.
"""
import argparse, json, os, warnings
warnings.filterwarnings("ignore")
import arviz as az


def diagnose(path):
    idata = az.from_netcdf(path)
    mu = [v for v in idata.posterior.data_vars if v.endswith("_mu")]
    s = az.summary(idata, var_names=mu)
    div = (int(idata.sample_stats["diverging"].values.sum())
           if "diverging" in idata.sample_stats else -1)
    rec = dict(max_rhat_mu=round(float(s["r_hat"].max()), 3),
               min_ess_mu=int(s["ess_bulk"].min()), div=div)
    if "log_likelihood" in idata.groups():
        try:
            lo = az.loo(idata, pointwise=True)
            rec.update(elpd_loo=round(float(lo.elpd_loo), 1),
                       p_loo=round(float(lo.p_loo), 1),
                       bad_k=int((lo.pareto_k.values > 0.7).sum()),
                       n_obs=int(lo.pareto_k.size))
        except Exception as e:
            rec["loo_err"] = str(e)[:80]
    else:
        rec["loo"] = "no log_likelihood"
    return idata, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", default="/shares/zne.uzh/gdehol/dyscalculic_ddm")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--compare", action="store_true",
                   help="also az.compare the labels that carry log_likelihood")
    p.add_argument("--json-out", default=None)
    a = p.parse_args()

    out, idatas = {}, {}
    for lab in a.labels:
        path = os.path.join(a.trace_dir, f"model-{lab}_trace.nc")
        idata, rec = diagnose(path)
        out[lab] = rec
        idatas[lab] = idata
        print(lab, json.dumps(rec))

    if a.compare:
        cmp_set = {l: idatas[l] for l in a.labels
                   if "log_likelihood" in idatas[l].groups()}
        if len(cmp_set) >= 2:
            print("\n=== az.compare (LOO) ===")
            print(az.compare(cmp_set, ic="loo").to_string())

    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        json.dump(out, open(a.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
