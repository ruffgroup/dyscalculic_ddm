"""Within-model recovery for the STATIC choice model.

The honest identifiability test the cross-model fit could not give: generate
choices *from the choice model itself* with a known group difference in EITHER
perceptual-only OR memory-only encoding noise, then refit the choice regression
and ask whether the contrast lands in the component it was generated in. Repeat
over several simulation seeds to see the recovery distribution (not one draw).

We reuse the REAL paradigm (real subjects / n1 / n2 / group sizes) so the design
— trial counts, stimulus distribution, 33-vs-33 groups — matches the data
exactly; only the choices are synthetic.

Run:
  python recover_choice.py --scenario perceptual --reps 4
  python recover_choice.py --scenario memory     --reps 4
"""
import argparse
import os.path as op

import numpy as np
import pandas as pd

from bauer.models import (MagnitudeComparisonModel,
                          MagnitudeComparisonRegressionModel)

HERE = op.dirname(op.abspath(__file__))

# Generating noise (log-magnitude SD units). Baseline shared by both groups;
# the deficit doubles ONE component in the dyscalc group, the other is held
# fixed. Magnitudes chosen near the real estimates so accuracy/compression are
# realistic and the effect is detectable but not trivial.
# Calibrated so control acc ≈ 0.875 (real control) and each deficit drops
# accuracy to ≈ 0.85 (real dyscalc, −0.025) — the REALISTIC effect size, via
# perceptual-only or memory-only, both in the small-noise regime.
BASE_PERC, BASE_MEM = 0.08, 0.13
DEF_PERC, DEF_MEM = 0.12, 0.20


def real_paradigm():
    """Real (subject, n1, n2, group) design; drop rt/choice, keep structure."""
    csv = op.join(HERE, 'data', 'magjudge_behavior_DNumRisk.csv')
    df = pd.read_csv(csv)
    keep = ['subject', 'n1', 'n2', 'group']
    df = df[keep].dropna().copy()
    df['subject'] = df['subject'].astype(int)
    # one trial index per subject
    df['trial_nr'] = df.groupby('subject').cumcount()
    return df.set_index(['subject', 'trial_nr'])


def generating_params(paradigm, scenario, seed):
    """Per-subject truth: double `scenario` noise in dyscalc, jitter slightly."""
    rng = np.random.default_rng(seed)
    sub_group = (paradigm.reset_index()[['subject', 'group']]
                 .drop_duplicates().set_index('subject')['group'])
    subs = sub_group.index
    perc, mem = {}, {}
    for s in subs:
        dys = sub_group[s] == 'dyscalc'
        perc[s] = (DEF_PERC if (dys and scenario == 'perceptual') else BASE_PERC)
        mem[s] = (DEF_MEM if (dys and scenario == 'memory') else BASE_MEM)
    return pd.DataFrame({
        'perceptual_noise_sd': [perc[s] * np.exp(rng.normal(0, 0.05)) for s in subs],
        'memory_noise_sd':     [mem[s] * np.exp(rng.normal(0, 0.05)) for s in subs],
    }, index=pd.Index(subs, name='subject'))


def simulate_choices(paradigm, par, seed):
    gen = MagnitudeComparisonModel(
        paradigm, fit_separate_evidence_sd=True,
        memory_model='shared_perceptual_noise')
    sim = gen.simulate(paradigm, par, n_samples=1)
    sim = sim.reset_index()
    sim['choice'] = sim['simulated_choice'].astype(bool)
    return sim


def fit_and_contrast(sim, draws, tune):
    sub_group = (sim[['subject', 'group']].drop_duplicates()
                 .set_index('subject')['group'])
    fit_df = sim[['subject', 'trial_nr', 'n1', 'n2', 'choice']].copy()
    fit_df['group'] = fit_df['subject'].map(sub_group)
    fit_df = fit_df.set_index(['subject', 'trial_nr'])

    regressors = {'perceptual_noise_sd': 'C(group)',
                  'memory_noise_sd': 'C(group)'}
    m = MagnitudeComparisonRegressionModel(
        fit_df, regressors, fit_separate_evidence_sd=True,
        memory_model='shared_perceptual_noise')
    m.build_estimation_model(fit_df, hierarchical=True)
    idata = m.sample(draws=draws, tune=tune, chains=4,
                     target_accept=0.95, backend='numpyro')

    post = idata.posterior

    def contrast(name):
        mu = post[f'{name}_mu']                 # (chain, draw, regressor)
        slope = mu.isel({mu.dims[-1]: 1}).values.ravel()  # C(group)[dyscalc]
        return slope.mean(), float((slope > 0).mean())

    import arviz as az
    rhat = float(az.summary(idata)['r_hat'].max())
    return {'perc': contrast('perceptual_noise_sd'),
            'mem': contrast('memory_noise_sd'), 'rhat': rhat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=['perceptual', 'memory'], required=True)
    ap.add_argument('--reps', type=int, default=4)
    ap.add_argument('--draws', type=int, default=600)
    ap.add_argument('--tune', type=int, default=600)
    args = ap.parse_args()

    paradigm = real_paradigm()
    print(f'{"="*64}\nCHOICE->CHOICE RECOVERY  |  truth = {args.scenario}-only '
          f'(dyscalc doubles {args.scenario} noise)\n{"="*64}')
    print(f'paradigm: {paradigm.reset_index().subject.nunique()} subjects, '
          f'{len(paradigm)} trials\n')

    rows = []
    for rep in range(args.reps):
        # truth params re-jittered + choices re-drawn per rep -> recovery spread
        par = generating_params(paradigm, args.scenario, seed=1000 + rep)
        sim = simulate_choices(paradigm, par, seed=2000 + rep)
        acc = (sim['choice'] == (sim['n2'] > sim['n1'])).mean()
        r = fit_and_contrast(sim, args.draws, args.tune)
        pc, pp = r['perc']; mc, mp = r['mem']
        rows.append({'rep': rep, 'acc': acc, 'perc_d': pc, 'perc_P': pp,
                     'mem_d': mc, 'mem_P': mp, 'rhat': r['rhat']})
        print(f'rep {rep}: acc={acc:.3f}  '
              f'perceptual Δ={pc:+.3f}(P={pp:.2f})  '
              f'memory Δ={mc:+.3f}(P={mp:.2f})  rhat={r["rhat"]:.2f}')

    res = pd.DataFrame(rows)
    print(f'\n{"-"*64}\nSUMMARY  (truth = {args.scenario}-only; recovered should '
          f'load on {args.scenario}, ~0 on the other)')
    print(f'  perceptual Δ: mean {res.perc_d.mean():+.3f}  '
          f'[{res.perc_d.min():+.3f}, {res.perc_d.max():+.3f}]  '
          f'median P(>0)={res.perc_P.median():.2f}')
    print(f'  memory     Δ: mean {res.mem_d.mean():+.3f}  '
          f'[{res.mem_d.min():+.3f}, {res.mem_d.max():+.3f}]  '
          f'median P(>0)={res.mem_P.median():.2f}')
    out = op.join(HERE, 'results', f'recover_choice_{args.scenario}.csv')
    import os
    os.makedirs(op.dirname(out), exist_ok=True)
    res.to_csv(out, index=False)
    print(f'  saved -> {out}')


if __name__ == '__main__':
    main()
