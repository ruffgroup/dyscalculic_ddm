# How to run a parameter-recovery check

Parametric-bootstrap recovery answers "if the truth were our posterior, would we
get it back?". It has **not** been run for the RDM (the model we report), so the
perceptual/memory split is currently model-based, not recovery-validated. This is
the outline to run it. Harness: `recover_bootstrap.py` (one rep), `run_recover.sh`
(SLURM array), `aggregate_recovery.py` (the cloud figure).

## The idea (one rep)

1. Draw one "plausible truth" — a single posterior draw `k` from the model's
   **real-data** posterior trace.
2. **Simulate** one dataset on the real paradigm from draw `k` (same n1/n2 design,
   same subjects).
3. **Refit** the same model class as a group regression.
4. Record **true vs recovered** group contrast (`C(group)[dyscalc]` slope, link
   scale) for each parameter, plus `max_rhat`.

Repeat ~100× → a cross-recovery cloud. **Diagonal** = self-recovery (slope→1,
bias→0, tight = recoverable). **Off-diagonal** = leakage (r→0 = the two
components are cleanly separated; large |r| = they trade off).

## Prerequisites

- The real-data posterior trace for the model, on the cluster:
  `/shares/zne.uzh/gdehol/dyscalculic_ddm/model-<model>_group_trace.nc`.
- The `bauer_cuda` env (bauer **v0.4.0**) on the cluster.
- Model labels the harness supports (`SPEC` in `recover_bootstrap.py`):
  `choice`, `choice_klw`, `ddm`, `ddm_sv`, `rdm`.

## Step 1 — simulate + refit the reps (cluster, SLURM array)

One array task = one rep → `results/recover_boot/<model>_rep<NN>.csv`.

```bash
# RDM (GPU; the one we actually need). Throttle concurrency with %16.
sbatch --array=0-99%16 --gres=gpu:L4:1 --job-name=dys_rec_rdm \
    run_recover.sh rdm /shares/zne.uzh/gdehol/dyscalculic_ddm/model-rdm_group_trace.nc

# Static choice is cheap (CPU, no GPU):
sbatch --array=0-99 --job-name=dys_rec_choice \
    run_recover.sh choice /shares/zne.uzh/gdehol/dyscalculic_ddm/model-choice_group_trace.nc
```

- Defaults per rep: `--draws 800 --tune 800 --chains 4`. Append overrides after
  the trace path, e.g. `... run_recover.sh rdm <trace> --tune 2000 --draws 1000`.
- Logs: `~/logs/<jobname>_<arrayjob>_<rep>.txt`.
- This uses the **current recipe** (5% `p_outlier` contaminant + HalfNormal
  priors) because it goes through `fit_ddm.build_model` — no extra flags needed.

## Step 2 — pull the reps back (or aggregate on the cluster)

```bash
rsync -a sciencecluster:git/dyscalculic_ddm/results/recover_boot/ results/recover_boot/
```

## Step 3 — aggregate into the recovery cloud

```bash
python aggregate_recovery.py --models rdm        # -> notes/figures/recovery_cloud.pdf
# several at once: --models choice ddm rdm
```

It keeps only converged reps (`max_rhat ≤ 1.05`, tune with `--rhat-max`), prints
`n_converged / n_total` and the per-component slope / bias / leak, and writes the
2×2 cross-recovery matrix per model to `notes/figures/recovery_cloud.pdf`.

## Reading the result

- **Self-recovery (diagonal):** slope ≈ 1, bias ≈ 0, tight cloud → the parameter
  is recoverable from this design.
- **Leakage (off-diagonal):** r ≈ 0 → clean separation; large |r| → the two
  encoding components trade off (the perceptual↔memory question).
- **`n_converged`:** low → the model doesn't sample well at these chain lengths;
  bump `--tune` / `--draws` and note the cost.

## Gotchas

- Old `results/recover_boot/{choice,ddm}_rep*.csv` predate the contaminant /
  HalfNormal fixes (when the DDM mixed badly); regenerate rather than trust them.
- `aggregate_recovery.py`'s pretty-name map has no `rdm` entry — it falls back to
  the raw label `rdm`. Cosmetic only.
- Each RDM rep costs roughly a real fit; budget GPU time and throttle the array.
