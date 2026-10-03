"""Targeted R2 ablations of R1 protection, NOT a new validity theorem.

P: original fitted conditional inference without bootstrap maximization.
K: protect only the calibration-derived common/contrast scale ratio while
holding target contrast-scale, radial shape and study shape at the TRAIN fit.
Both retain the existing nonnegative projection, pilot family and learned gate.
Neither estimated frontend has a general finite-sample input-validity proof.
"""
from copy import deepcopy
import numpy as np
from .robustness_bootstrap_guard import nuisance, learn_profiles, components, MODES, MULTIPLIERS
from .robustness_methods import contrasts, fit_calibration
from .robustness_pilot_selection import pilot_training_receipt, apply_pilot_multiplier


def restore_theta(value):
    theta = deepcopy(value)
    theta['shape'] = np.asarray(theta['shape'], float)
    if theta['prior']['gaussian_bic_selected']:
        theta['prior']['df'] = np.inf
    elif theta['prior']['df'] is None:
        raise ValueError('Missing non-Gaussian radial df')
    return theta


def rho_guard_theta(original, rho_values):
    """Preserve eta²=(1-rho)*scatter and all target shape parameters.

    kappa(rho) is increasing on the admissible compound-correlation interval.
    For fixed target fit, directions and a positive statistic, the largest rho
    gives the largest normal/Student p-value. This is an algebraic maximum,
    not a confidence statement that rho_star bounds the true rho.
    """
    rho = np.asarray(rho_values, float)
    k = original['dimension']//4 + 1
    if rho.ndim != 1 or not len(rho) or not np.isfinite(rho).all() or np.any(rho <= -1/(k-1)) or np.any(rho >= 1):
        raise ValueError('Finite admissible calibration rho values required')
    star = max(float(original['rho']), float(rho.max()))
    result = restore_theta(original)
    if star != original['rho']:
        result['prior']['scatter'] *= (1-original['rho'])/(1-star)
        result['rho'] = star
    return result


def calibration_rho_bank(original, calibration, n_train, draws, rng):
    rhos = [original['rho']]
    receipts = []
    for draw in range(draws):
        # Keep historical R1 random streams paired; these gene indices are
        # discarded, not used to refit target noise or to learn a new profile.
        rng.integers(0, n_train, n_train)
        rows = rng.integers(0, len(calibration), len(calibration))
        try:
            fitted = fit_calibration(calibration[rows])
            if not fitted.converged:
                raise ArithmeticError('Calibration optimizer did not converge')
            rhos.append(fitted.rho)
            receipts.append({'draw': draw, 'status': 'completed', 'calibration_indices': rows.tolist(),
                             'rho': fitted.rho, 'fit': vars(fitted)})
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            receipts.append({'draw': draw, 'status': 'failed', 'calibration_indices': rows.tolist(), 'error': repr(error)})
            return rhos, receipts, False
    return rhos, receipts, True


def evaluate(z, calibration, *, seed=4151601, draws=16, mode='both', cached_folds=None):
    """Return all equal-frontend comparators; truth is not an input.

    cached_folds is a paired DEVELOPMENT accelerator, not deployed fitting:
    the caller must verify frozen source, input and old record hashes. Only
    fully successful old banks are reusable for K. Independent fresh-refit
    parity tests are required. Cached elapsed time is not deployment speed.
    """
    z, cal = np.asarray(z, float), np.asarray(calibration, float)
    if (mode not in ('P', 'K', 'both') or z.ndim != 3 or z.shape[1] != 4 or z.shape[0] < 8
            or z.shape[2] < 2 or cal.ndim != 3 or cal.shape[0] != 4 or cal.shape[2] != z.shape[2]
            or not np.isfinite(z).all() or not np.isfinite(cal).all() or draws < 1):
        raise ValueError('Finite matching Gx4xK inputs and positive draw count required')
    if cached_folds is not None and len(cached_folds) != 2:
        raise ValueError('Exactly two source-verified cached folds required')
    tags = ['P', 'K'] if mode == 'both' else [mode]
    g, _, k = z.shape
    means = z.mean(-1)
    residual = (z-means[..., None])@contrasts(k)
    flat = cal.reshape(-1, k)
    output = {f'R2{tag}_pilotc{c}_{kernel}_eBH': np.empty((g, 2))
              for tag in tags for c in MULTIPLIERS for kernel in [*MODES, 'projection_gate']}
    pvalues = {f'{tag}_{kernel}': np.empty((g, 2)) for tag in tags for kernel in MODES}
    folds = []
    for fold in range(2):
        held = np.arange(g)%2 == fold
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
        if cached_folds is None:
            original = nuisance(residual[~held], flat)
            profiles, gamma, pattern = learn_profiles(means[~held], residual[~held], original)
        else:
            cached = cached_folds[fold]
            if cached['fold'] != fold or cached['training_genes'] != int((~held).sum()):
                raise ValueError('Cached fold identity mismatch')
            original = restore_theta(cached['bank'][0])
            profiles, gamma = np.asarray(cached['profiles']), np.asarray(cached['gamma'])
            pattern = deepcopy(cached['pattern'])
        success, receipts, rhos = True, [], [original['rho']]
        if 'K' in tags:
            if cached_folds is None:
                rhos, receipts, success = calibration_rho_bank(original, flat, int((~held).sum()), draws, rng)
            else:
                if not cached['guard_success'] or len(cached['bank']) != draws+1:
                    raise ValueError('Incomplete original bank needs fresh refit, not guessed cached rho')
                rhos = [theta['rho'] for theta in cached['bank']]
                receipts = [{'draw': r['draw'], 'status': r['status'], 'calibration_indices': r['calibration_indices']}
                            for r in cached['bootstrap_receipts']]
        guarded = rho_guard_theta(original, rhos)
        bank = [original, guarded] if 'K' in tags else [original]
        training = components(means[~held], residual[~held], profiles, original, bank)
        actual = components(means[held], residual[held], profiles, original, bank)
        if not success:
            for kernel in MODES:
                training['guard'][kernel][:] = 1.
                actual['guard'][kernel][:] = 1.
        pilots = {}
        for tag in tags:
            original_tag = 'plugin' if tag == 'P' else 'guard'
            for kernel in MODES:
                pilot = pilot_training_receipt(training[original_tag][kernel])
                pilots[tag+'_'+kernel] = pilot
                pvalues[tag+'_'+kernel][held] = actual[original_tag][kernel]
                for c in MULTIPLIERS:
                    output[f'R2{tag}_pilotc{c}_{kernel}_eBH'][held] = apply_pilot_multiplier(
                        actual[original_tag][kernel], pilot, 2*g, c)[0]
            for c in MULTIPLIERS:
                output[f'R2{tag}_pilotc{c}_projection_gate_eBH'][held] = (
                    gamma*output[f'R2{tag}_pilotc{c}_projection_eBH'][held]
                    +(1-gamma)*output[f'R2{tag}_pilotc{c}_ordinary_bonf_eBH'][held])
        folds.append({'fold': fold, 'training_genes': int((~held).sum()),
                      'original': original, 'guarded': guarded, 'rho_bank': rhos,
                      'calibration_receipts': receipts, 'K_success': success,
                      'profiles': profiles, 'gamma': gamma, 'pattern': pattern, 'pilot': pilots})
    return output, {'folds': folds, 'held_pvalues': pvalues, 'cached_development_only': cached_folds is not None,
                    'method_version': 'R2-protection-efficiency-dev1', 'uses_truth': False,
                    'validity': 'EMPIRICAL; calibration-rho maximum has no parameter coverage theorem',
                    'bootstrap_draws': draws, 'bootstrap_seed': seed, 'mode': mode}
