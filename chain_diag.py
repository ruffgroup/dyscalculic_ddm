"""Is one chain the culprit? Per-chain means + leave-one-chain-out r-hat/ESS
for the group-level (*_mu) parameters. If dropping a single chain collapses
r-hat to ~1.0 and ESS jumps, that chain is stuck and the fit is salvageable.

  python chain_diag.py --labels rdm_all3_cont_hn rdm_pm_cont_hn
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")
import arviz as az, numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", default="/shares/zne.uzh/gdehol/dyscalculic_ddm")
    p.add_argument("--labels", nargs="+", required=True)
    a = p.parse_args()
    for lab in a.labels:
        idata = az.from_netcdf(os.path.join(a.trace_dir, f"model-{lab}_trace.nc"))
        post = idata.posterior
        mu = [v for v in post.data_vars if v.endswith("_mu")]
        nchains = post.sizes["chain"]
        print(f"\n=== {lab}  ({nchains} chains) ===")

        # worst-mixing mu param overall
        summ = az.summary(idata, var_names=mu)
        worst = summ["r_hat"].idxmax()
        print(f"worst mu param: {worst}  (full r_hat={summ.loc[worst,'r_hat']:.3f}, "
              f"min ESS_bulk over mu={summ['ess_bulk'].min():.0f})")

        # per-chain mean of that param (flatten any regressor dim)
        da = post[worst.split('[')[0]] if '[' in worst else post[worst]
        flat = da.stack(s=[d for d in da.dims if d not in ('chain',)])
        pcm = flat.mean('s').values  # per-chain mean
        print("per-chain mean of worst param:",
              ", ".join(f"c{i}={v:.3f}" for i, v in enumerate(pcm)))

        # leave-one-chain-out: max r_hat over all mu params
        full_rhat = float(summ["r_hat"].max())
        print(f"max r_hat over all mu (all {nchains} chains): {full_rhat:.3f}")
        for drop in range(nchains):
            keep = [c for c in range(nchains) if c != drop]
            sub = idata.sel(chain=keep)
            s = az.summary(sub, var_names=mu)
            print(f"  drop chain {drop}: max r_hat={s['r_hat'].max():.3f}  "
                  f"min ESS_bulk={s['ess_bulk'].min():.0f}")


if __name__ == "__main__":
    main()
