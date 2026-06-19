"""Read out a fitted DDM trace: convergence + (for ddm_group) group contrasts.

    ~/mambaforge/envs/bauer/bin/python analyze_fit.py model-ddm_group_trace.nc

For the regression model (ddm_group) the interesting quantities are the
*group-level* regression coefficients on the non-Intercept regressor
(`group[T.dyscalc]` or `group`): a coefficient whose 94% HDI excludes 0 means
that parameter differs between dyscalculics and controls. Noise SDs are fit on
a pre-softplus scale, so the coefficient *sign* is interpretable (positive =
larger in dyscalc) even though its magnitude isn't the raw SD difference.
"""
import argparse

import arviz as az
import numpy as np


def main(nc):
    idata = az.from_netcdf(nc)

    summ = az.summary(idata)
    n_div = int(idata.sample_stats["diverging"].sum()) \
        if "diverging" in idata.sample_stats else -1
    print(f"=== {nc} ===")
    print(f"max r_hat={summ['r_hat'].max():.4f}  "
          f"min ess_bulk={summ['ess_bulk'].min():.0f}  "
          f"divergences={n_div}")

    post = idata.posterior
    # Group-level regression coefficients live in vars like
    # 'perceptual_noise_sd_mu' with a '*_regressors' dim.
    reg_vars = [v for v in post.data_vars
                if v.endswith("_mu") and any(d.endswith("_regressors")
                                             for d in post[v].dims)]
    if not reg_vars:
        print("\n(no regression coefficients — looks like a baseline fit)")
        return

    print("\n=== Group-level coefficients (94% HDI; * = HDI excludes 0) ===")
    for v in sorted(reg_vars):
        rdim = next(d for d in post[v].dims if d.endswith("_regressors"))
        for lvl in post.coords[rdim].values:
            draws = post[v].sel({rdim: lvl}).values.ravel()
            lo, hi = az.hdi(draws, hdi_prob=0.94)
            p_pos = float((draws > 0).mean())
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"{star} {v:<28} {str(lvl):<22} "
                  f"mean={draws.mean():+.3f}  HDI=[{lo:+.3f}, {hi:+.3f}]  "
                  f"P(>0)={p_pos:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("nc", help="path to a *_trace.nc file")
    args = p.parse_args()
    main(args.nc)
