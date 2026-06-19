# Brief for Maike — dyscalculic_ddm

Hi Maike — this is the magnitude-comparison modelling project (DNumRisk: 66
subjects, 33 control / 33 dyscalculic, "see n1, then n2, choose the larger").
Everything is on GitHub at **`github.com/ruffgroup/dyscalculic_ddm`**. This page
gets you to the result fast.

## The one thing to look at: `notes/figures/overview_verdict.pdf`

That 10-page PDF **is** the result — it's committed, so you have it the moment
you clone. To regenerate it yourself (no cluster, no big files — the group-level
summary it's built from is committed):

```bash
python plot_overview.py     # reads notes/data/overview.{json,npz}
                            # writes notes/figures/overview_verdict.pdf
```

Read [`CONCLUSIONS.md`](CONCLUSIONS.md) alongside it (the verdict, with numbers)
and [`MODELS.md`](MODELS.md) (what each model in the figure is).

## The story (and it's robust — it improves on the older draft)

We model the decision-maker as repeatedly sampling a **perceptual posterior**
each time step, so noisy perception lands in the **drift rate** (hard trials are
slower). The model that fits is the **advantage race diffusion** (van
Ravenzwaaij 2020), not the DDM — the DDM can't reproduce the magnitude effect
and loses on LOO by ~730 (figure p8–10). On the RDM:

- **Dyscalculics have a noisier number representation** (P = 1.00). Perceptual
  noise 0.34 → 0.45; **memory noise roughly doubles, 0.049 → 0.091** — the held
  first number is hit hardest.
- **They compensate with a higher decision threshold** (0.50 → 0.65, P = 1.00).
  That's why in raw behaviour they're clearly slower but only *slightly* less
  accurate (calibrating to a target accuracy).
- **Higher urgency** (`w_0` 0.31 → 0.40) and a **slightly larger magnitude
  effect** (`w_s`), and a **larger compression effect** (more regression to the
  prior, because they're noisier).

Headline in one line: **noisier → slower, with stronger compression, especially
in memory.**

## Parameter recovery (to-do)

We haven't run a recovery check for the RDM yet, so the perceptual/memory split
is model-based, not recovery-validated. The harness is in the repo and
[`recovery_howto.md`](recovery_howto.md) is a step-by-step outline for running it
(a cluster array via `run_recover.sh`, then `aggregate_recovery.py`). Older
choice/DDM reps predate the contaminant/HalfNormal fixes — back when the DDM
mixed badly — and aren't shipped.

## Getting set up

```bash
git clone git@github.com:ruffgroup/dyscalculic_ddm.git
cd dyscalculic_ddm
conda env create -f environment.yml      # installs the exact bauer (see below)
conda activate dyscalculic_ddm
```

The **behavioural data** is human-subjects data and is not in the repo — ask
Gilles for `data/magjudge_behavior_DNumRisk.csv` and drop it into `data/`. The
41 GB of posterior traces are only needed if you want to *refit* (they live on
the cluster: `rsync -a sciencecluster:/shares/zne.uzh/gdehol/dyscalculic_ddm/ results/`).

## bauer version — use exactly this

The models live in **bauer**, pinned in `environment.yml` to the immutable tag
**`v0.4.0`** (= `github.com/ruffgroup/bauer` @ `e406694`). Don't substitute
`v0.3.0` or `main` — the 0.4.0 line is on a divergent branch and anything else
gives subtly different model code. The env file handles it; to install by hand:
`pip install "git+https://github.com/ruffgroup/bauer.git@v0.4.0"`.

The hard-won fitting gotchas (contaminant model, HalfNormal priors, fixing
`w_d`) that make these models converge at all are written up in
[`../CLAUDE.md`](../CLAUDE.md) — worth a read before you refit anything.

Shout if anything doesn't run — happy to walk through it.
