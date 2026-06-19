"""Validation for the memory_as_sv DDM variant.

(1) Slow errors: simulate from old (memory→SNR) vs new (memory→sv) model with
    the SAME generative noise, show new model produces slow errors.
(2) Parameter recovery: simulate two groups differing in EITHER perceptual-only
    OR memory-only, refit with the memory_as_sv regression model, check the
    group contrast lands in the right parameter.
"""
import sys
import numpy as np
import pandas as pd
import arviz as az

from bauer.models.ddm import (DDMMagnitudeComparisonModel,
                              DDMMagnitudeComparisonRegressionModel)


def make_paradigm(subjects, n_per_subj=300, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([5, 7, 10, 14, 20, 28])
    rows = []
    for s in subjects:
        n1 = rng.choice(base, n_per_subj)
        frac = np.exp(rng.uniform(np.log(0.4), np.log(2.5), n_per_subj))
        n2 = np.clip((n1 * frac).round(), 2, None)
        rows.append(pd.DataFrame({'subject': s, 'trial_nr': np.arange(n_per_subj),
                                  'n1': n1, 'n2': n2}))
    return pd.concat(rows).set_index(['subject', 'trial_nr'])


def simulate_dataset(model, paradigm, par_df, seed):
    """One faithful simulated dataset (one (rt, choice) per trial)."""
    sim = model.simulate(paradigm, par_df, n_samples=1, random_seed=seed)
    sim = sim.reset_index()
    sim['choice'] = sim['simulated_choice'].astype(bool)
    sim['rt'] = sim['simulated_rt'].astype(float)
    sim['correct'] = sim['choice'] == (sim['n2'] > sim['n1'])
    return sim


def quantile_table(sim, qs=(0.1, 0.3, 0.5, 0.7, 0.9)):
    out = {}
    for label, mask in [('correct', sim['correct']), ('error', ~sim['correct'])]:
        rts = sim.loc[mask, 'rt']
        out[label] = np.quantile(rts, qs) if len(rts) > 5 else np.full(len(qs), np.nan)
    return pd.DataFrame(out, index=[f'q{int(q*100)}' for q in qs])


def slow_error_check():
    print('=' * 60)
    print('(1) SLOW-ERROR SIGNATURE (sv on vs sv off, matched drift)')
    print('=' * 60)
    subs = [1, 2, 3, 4]
    paradigm = make_paradigm(subs, n_per_subj=600, seed=1)
    # Use the memory_as_sv model for BOTH; for the "no sv" reference we zero out
    # memory noise and inflate perceptual to match overall accuracy, so the only
    # difference is whether the drift varies across trials (sv) or not. This
    # isolates the Ratcliff slow-error mechanism rather than a drift-magnitude
    # confound.
    par_sv = pd.DataFrame({
        'perceptual_noise_sd': 0.22, 'memory_noise_sd': 0.7,
        'a': 1.4, 't0': 0.3,
    }, index=pd.Index(subs, name='subject'))
    # No-sv: route the memory variance into perceptual (within-trial) instead,
    # giving comparable accuracy but sv=0.
    par_nosv = pd.DataFrame({
        'perceptual_noise_sd': 0.40, 'memory_noise_sd': 0.0,
        'a': 1.4, 't0': 0.3,
    }, index=pd.Index(subs, name='subject'))

    m = DDMMagnitudeComparisonModel(
        paradigm, fit_separate_evidence_sd=True,
        memory_model='shared_perceptual_noise', memory_as_sv=True)

    for name, par in [('NO sv  (memory=0, perceptual inflated)', par_nosv),
                      ('WITH sv (memory->sv)', par_sv)]:
        sim = simulate_dataset(m, paradigm, par, seed=42)
        qt = quantile_table(sim)
        acc = sim['correct'].mean()
        slow = qt.loc['q50', 'error'] - qt.loc['q50', 'correct']
        slow90 = qt.loc['q90', 'error'] - qt.loc['q90', 'correct']
        print(f'\n{name}:  acc={acc:.3f}')
        print(qt.round(3))
        print(f'  error-correct median delta : {slow:+.3f} s')
        print(f'  error-correct q90    delta : {slow90:+.3f} s')


def recovery_check(scenario, draws=600, tune=600):
    print('=' * 60)
    print(f'(3) PARAMETER RECOVERY: group differs in {scenario}-only')
    print('=' * 60)
    # 6 control + 6 patient subjects.
    controls = list(range(1, 7))
    patients = list(range(7, 13))
    subs = controls + patients
    paradigm = make_paradigm(subs, n_per_subj=400, seed=7)

    # Baseline params shared; group difference injected only into one component.
    rng = np.random.default_rng(123)
    perc = {s: 0.25 for s in subs}
    mem = {s: 0.45 for s in subs}
    if scenario == 'perceptual':
        for s in patients:
            perc[s] = 0.50          # doubled perceptual in patients
    elif scenario == 'memory':
        for s in patients:
            mem[s] = 0.90           # doubled memory in patients
    # small per-subject jitter so it's not perfectly degenerate
    par = pd.DataFrame({
        'perceptual_noise_sd': [perc[s] * np.exp(rng.normal(0, 0.05)) for s in subs],
        'memory_noise_sd':     [mem[s] * np.exp(rng.normal(0, 0.05)) for s in subs],
        'a': 1.4, 't0': 0.3,
    }, index=pd.Index(subs, name='subject'))

    gen = DDMMagnitudeComparisonModel(
        paradigm, fit_separate_evidence_sd=True,
        memory_model='shared_perceptual_noise', memory_as_sv=True)
    sim = simulate_dataset(gen, paradigm, par, seed=2024)
    sim['group'] = np.where(sim['subject'].isin(patients), 'patient', 'control')

    fit_df = sim[['subject', 'trial_nr', 'n1', 'n2', 'choice', 'rt', 'group']].copy()
    fit_df = fit_df.set_index(['subject', 'trial_nr'])

    regressors = {'perceptual_noise_sd': 'C(group)',
                  'memory_noise_sd': 'C(group)'}
    m = DDMMagnitudeComparisonRegressionModel(
        fit_df, regressors,
        fit_separate_evidence_sd=True,
        memory_model='shared_perceptual_noise', memory_as_sv=True)
    m.build_estimation_model(fit_df, hierarchical=True)
    idata = m.sample(draws=draws, tune=tune, chains=2,
                     target_accept=0.9, backend='numpyro')

    post = idata.posterior
    # group-level contrast = the C(group)[T.patient] slope mu.
    def contrast(par_name):
        mu = post[f'{par_name}_mu']
        # dim ordering: (chain, draw, <regressor>)
        slope = mu.isel({mu.dims[-1]: 1}).values.ravel()
        return slope.mean(), np.quantile(slope, [0.025, 0.975])

    pc_m, pc_ci = contrast('perceptual_noise_sd')
    mc_m, mc_ci = contrast('memory_noise_sd')
    print(f'\nTRUE group difference: {scenario}-only '
          f'(patients have larger {scenario} noise)')
    print(f'  perceptual C(group)[patient] slope: {pc_m:+.3f}  '
          f'95% CI [{pc_ci[0]:+.3f}, {pc_ci[1]:+.3f}]')
    print(f'  memory     C(group)[patient] slope: {mc_m:+.3f}  '
          f'95% CI [{mc_ci[0]:+.3f}, {mc_ci[1]:+.3f}]')
    print(f'  max r_hat: {float(az.summary(idata)["r_hat"].max()):.3f}')
    return idata


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('all', 'slow'):
        slow_error_check()
    if what in ('all', 'memory'):
        recovery_check('memory')
    if what in ('all', 'perceptual'):
        recovery_check('perceptual')
