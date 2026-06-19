"""Group-contrast posteriors (dyscalc - control) for fitted regression traces.

For each *_mu parameter that carries a group regressor, report the non-Intercept
coefficient's posterior mean / sd / 94% HDI / P(>0). These are the de-confounded
"where does the deficit sit" numbers. Coefficients are on each parameter's link
scale (noise SDs are positive via a log/softplus link, so a coefficient > 0 =
MORE noise in dyscalc).

  python group_contrasts.py --labels ddm_pm_cont ddm_group_cont \
      --trace-dir /shares/zne.uzh/gdehol/dyscalculic_ddm
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import arviz as az, numpy as np


def contrasts(idata):
    rows = []
    for v in idata.posterior.data_vars:
        if not v.endswith("_mu"):
            continue
        da = idata.posterior[v]
        rdim = [d for d in da.dims if d.endswith("_regressors")]
        if not rdim:
            continue
        rdim = rdim[0]
        for lv in da.coords[rdim].values:
            if lv == "Intercept":
                continue
            b = da.sel({rdim: lv}).values.ravel()
            lo, hi = az.hdi(b, hdi_prob=0.94)
            rows.append((v.replace("_mu", ""), str(lv), b.mean(), b.std(),
                         lo, hi, (b > 0).mean()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", default="/shares/zne.uzh/gdehol/dyscalculic_ddm")
    p.add_argument("--labels", nargs="+", required=True)
    a = p.parse_args()
    for lab in a.labels:
        idata = az.from_netcdf(os.path.join(a.trace_dir, f"model-{lab}_trace.nc"))
        print(f"\n=== {lab} ===")
        print(f"{'param':<22}{'level':<16}{'mean':>8}{'sd':>8}"
              f"{'hdi2.5':>9}{'hdi97.5':>9}{'P>0':>7}")
        for param, lv, m, s, lo, hi, pgt in contrasts(idata):
            print(f"{param:<22}{lv:<16}{m:>8.3f}{s:>8.3f}{lo:>9.3f}{hi:>9.3f}{pgt:>7.2f}")


if __name__ == "__main__":
    main()
