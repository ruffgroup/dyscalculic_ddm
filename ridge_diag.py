"""Where does the RDM mixing pathology actually live? Three probes:

1. ESS ranking of ALL parameters (group-mu vs group-sd vs subject-level), so we
   see whether it's the drift block, the non-decision time, or a hierarchical
   SD funnel that mixes worst.
2. Posterior correlation matrix among the group-level (*_mu) parameters — a
   ridge shows up as a near-±1 correlation between specific parameters. Reports
   the strongest off-diagonal pairs.
3. BFMI per chain (energy diagnostic): low BFMI => the sampler can't traverse
   the energy distribution (funnel / heavy tails), distinct from a flat ridge.

  python ridge_diag.py --label rdm_all3_cont_hn
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import arviz as az, numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", default="/shares/zne.uzh/gdehol/dyscalculic_ddm")
    p.add_argument("--label", required=True)
    a = p.parse_args()
    idata = az.from_netcdf(os.path.join(a.trace_dir, f"model-{a.label}_trace.nc"))
    post = idata.posterior
    print(f"=== {a.label} ===")

    # ---- 1. ESS ranking across ALL params ----
    s = az.summary(idata)
    s["kind"] = np.where(s.index.str.contains("_mu"), "group_mu",
                np.where(s.index.str.contains("_sd"), "group_sd",
                np.where(s.index.str.contains("_offset"), "subj_offset", "other")))
    print("\n-- min ESS_bulk by parameter KIND --")
    print(s.groupby("kind")["ess_bulk"].agg(["min", "median", "count"]).round(0))
    print("\n-- 15 worst-mixing parameters (lowest ESS_bulk) --")
    print(s.sort_values("ess_bulk")[["ess_bulk", "r_hat", "kind"]].head(15).to_string())

    # ---- 2. posterior correlations among group means ----
    mu = [v for v in post.data_vars if v.endswith("_mu")]
    cols, names = [], []
    for v in mu:
        da = post[v]
        extra = [d for d in da.dims if d not in ("chain", "draw")]
        if not extra:
            cols.append(da.values.ravel()); names.append(v.replace("_mu", ""))
        else:
            for lv in da.coords[extra[0]].values:
                cols.append(da.sel({extra[0]: lv}).values.ravel())
                names.append(f"{v.replace('_mu','')}[{lv}]")
    X = np.array(cols)
    C = np.corrcoef(X)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((abs(C[i, j]), C[i, j], names[i], names[j]))
    pairs.sort(reverse=True)
    print("\n-- 12 strongest group-mean posterior correlations --")
    for ab, c, n1, n2 in pairs[:12]:
        print(f"  {c:+.3f}   {n1:<28} {n2}")

    # ---- 3. BFMI per chain ----
    bfmi = az.bfmi(idata)
    print("\n-- BFMI per chain (want > ~0.3; low => energy/funnel problem) --")
    print("  ", np.round(bfmi, 3))


if __name__ == "__main__":
    main()
