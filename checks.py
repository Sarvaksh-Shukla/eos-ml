"""
Validation and consistency checks for EoS inference results.
Implements physical and statistical validation of posterior samples.
"""

import numpy as np
import warnings
from scipy import stats
from scipy.stats import kstest, anderson

from ..eos.eos_param import PiecewisePolytrope, SpectralEoS
from ..eos.constraints import EoSConstraints, ObservationalConstraints
from ..physics.tov import TOVSolver

class PosteriorValidator:
    """Validation tools for posterior samples."""

    def __init__(self, samples, param_names=None):
        """
        Initialize validator with posterior samples.

        Parameters:
        -----------
        samples : array_like
            Posterior samples (n_samples, n_params)
        param_names : list, optional
            Parameter names
        """
        self.samples = np.array(samples)
        self.n_samples, self.n_params = self.samples.shape

        if param_names is None:
            self.param_names = [f'param_{i}' for i in range(self.n_params)]
        else:
            self.param_names = param_names

    def check_sample_quality(self):
        """Check basic quality of posterior samples."""
        results = {}

        # Check for NaN or infinite values
        has_nan = np.any(np.isnan(self.samples))
        has_inf = np.any(np.isinf(self.samples))

        results['has_nan'] = has_nan
        results['has_inf'] = has_inf
        results['valid_samples'] = not (has_nan or has_inf)

        # Check for sufficient samples
        results['n_samples'] = self.n_samples
        results['sufficient_samples'] = self.n_samples > 1000

        # Check parameter ranges
        param_ranges = []
        for i in range(self.n_params):
            param_min = np.min(self.samples[:, i])
            param_max = np.max(self.samples[:, i])
            param_std = np.std(self.samples[:, i])

            param_ranges.append({
                'min': param_min,
                'max': param_max,
                'std': param_std,
                'range': param_max - param_min
            })

        results['parameter_ranges'] = param_ranges

        return results

    def check_convergence(self, n_chains=None):
        """Check MCMC convergence using Gelman-Rubin diagnostic."""
        if n_chains is None:
            # Split samples into multiple chains
            n_chains = min(4, self.n_samples // 1000)
            if n_chains < 2:
                warnings.warn("Insufficient samples for convergence check")
                return {'converged': None, 'r_hat': None}

        chain_length = self.n_samples // n_chains
        chains = []

        for i in range(n_chains):
            start_idx = i * chain_length
            end_idx = (i + 1) * chain_length
            chains.append(self.samples[start_idx:end_idx])

        chains = np.array(chains)  # (n_chains, chain_length, n_params)

        # Calculate R-hat for each parameter
        r_hat_values = []

        for param_idx in range(self.n_params):
            param_chains = chains[:, :, param_idx]

            # Between-chain variance
            chain_means = np.mean(param_chains, axis=1)
            overall_mean = np.mean(chain_means)
            B = chain_length * np.var(chain_means, ddof=1)

            # Within-chain variance
            chain_vars = np.var(param_chains, axis=1, ddof=1)
            W = np.mean(chain_vars)

            # R-hat statistic
            if W > 0:
                var_plus = ((chain_length - 1) * W + B) / chain_length
                r_hat = np.sqrt(var_plus / W)
            else:
                r_hat = np.inf

            r_hat_values.append(r_hat)

        r_hat_values = np.array(r_hat_values)
        converged = np.all(r_hat_values < 1.1)  # Standard threshold

        return {
            'converged': converged,
            'r_hat': r_hat_values,
            'max_r_hat': np.max(r_hat_values),
            'param_names': self.param_names
        }

    def check_effective_sample_size(self):
        """Estimate effective sample size for each parameter."""
        ess_values = []

        for param_idx in range(self.n_params):
            param_samples = self.samples[:, param_idx]

            # Simple autocorrelation-based ESS estimate
            try:
                # Calculate autocorrelation
                def autocorr(x, max_lag=None):
                    if max_lag is None:
                        max_lag = len(x) // 4

                    autocorrs = []
                    for lag in range(max_lag):
                        if lag == 0:
                            autocorrs.append(1.0)
                        else:
                            c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                            if np.isnan(c):
                                c = 0.0
                            autocorrs.append(c)

                    return np.array(autocorrs)

                autocorrs = autocorr(param_samples)

                # Find first negative autocorrelation
                neg_idx = np.where(autocorrs < 0)[0]
                if len(neg_idx) > 0:
                    tau = 2 * np.sum(autocorrs[:neg_idx[0]]) - 1
                else:
                    tau = 2 * np.sum(autocorrs) - 1

                tau = max(tau, 1.0)  # Ensure tau >= 1
                ess = self.n_samples / (2 * tau + 1)

            except Exception:
                ess = self.n_samples  # Fallback

            ess_values.append(ess)

        return {
            'ess': np.array(ess_values),
            'min_ess': np.min(ess_values),
            'adequate_ess': np.all(np.array(ess_values) > 100)
        }

def validate_eos_parameters(samples, eos_type='piecewise_polytrope', 
                          param_names=None, verbose=True):
    """
    Comprehensive validation of EoS parameter posterior samples.

    Parameters:
    -----------
    samples : array_like
        Posterior samples
    eos_type : str
        Type of EoS parameterization
    param_names : list, optional
        Parameter names
    verbose : bool
        Print validation results

    Returns:
    --------
    dict : Validation results
    """
    if param_names is None:
        if eos_type == 'piecewise_polytrope':
            param_names = ['log_p1', 'gamma1', 'gamma2', 'gamma3']
        else:
            param_names = [f'param_{i}' for i in range(samples.shape[1])]

    validator = PosteriorValidator(samples, param_names)
    results = {}

    if verbose:
        print("=== EoS Parameter Validation ===")

    # Basic sample quality
    quality_results = validator.check_sample_quality()
    results['quality'] = quality_results

    if verbose:
        print(f"Sample quality:")
        print(f"  Valid samples: {'✓' if quality_results['valid_samples'] else '✗'}")
        print(f"  Number of samples: {quality_results['n_samples']}")
        print(f"  Sufficient samples: {'✓' if quality_results['sufficient_samples'] else '✗'}")

    # Convergence check
    convergence_results = validator.check_convergence()
    results['convergence'] = convergence_results

    if verbose and convergence_results['converged'] is not None:
        print(f"Convergence (R-hat):")
        print(f"  Converged: {'✓' if convergence_results['converged'] else '✗'}")
        print(f"  Max R-hat: {convergence_results['max_r_hat']:.3f}")

    # Effective sample size
    ess_results = validator.check_effective_sample_size()
    results['ess'] = ess_results

    if verbose:
        print(f"Effective sample size:")
        print(f"  Min ESS: {ess_results['min_ess']:.0f}")
        print(f"  Adequate ESS: {'✓' if ess_results['adequate_ess'] else '✗'}")

    # Physical constraint validation
    physical_results = validate_physical_constraints(samples, eos_type, verbose=verbose)
    results['physical'] = physical_results

    # Overall validation
    overall_valid = (
        quality_results['valid_samples'] and
        quality_results['sufficient_samples'] and
        (convergence_results['converged'] is None or convergence_results['converged']) and
        ess_results['adequate_ess'] and
        physical_results['fraction_valid'] > 0.8
    )

    results['overall_valid'] = overall_valid

    if verbose:
        print(f"Overall validation: {'✓' if overall_valid else '✗'}")
        print("=" * 35)

    return results

def validate_physical_constraints(samples, eos_type='piecewise_polytrope', 
                                n_check=100, verbose=True):
    """
    Validate physical constraints for posterior samples.

    Parameters:
    -----------
    samples : array_like
        Posterior samples
    eos_type : str
        Type of EoS parameterization
    n_check : int
        Number of samples to check (for computational efficiency)
    verbose : bool
        Print results

    Returns:
    --------
    dict : Physical constraint validation results
    """
    n_samples = len(samples)
    n_check = min(n_check, n_samples)

    # Sample subset for checking
    check_indices = np.random.choice(n_samples, n_check, replace=False)
    check_samples = samples[check_indices]

    valid_count = 0
    constraint_violations = {
        'causality': 0,
        'stability': 0,
        'low_density': 0,
        'high_density': 0,
        'tov_failure': 0
    }

    mass_radius_data = []

    for i, params in enumerate(check_samples):
        try:
            # Create EoS
            if eos_type == 'piecewise_polytrope':
                eos = PiecewisePolytrope(*params)
            elif eos_type == 'spectral':
                eos = SpectralEoS(params)
            else:
                continue

            # Check physical constraints
            constraint_results = EoSConstraints.check_all_constraints(eos)

            if not constraint_results['causality']['satisfied']:
                constraint_violations['causality'] += 1
                continue

            if not constraint_results['stability']['satisfied']:
                constraint_violations['stability'] += 1
                continue

            if not constraint_results['low_density']['satisfied']:
                constraint_violations['low_density'] += 1
                continue

            if not constraint_results['high_density']['satisfied']:
                constraint_violations['high_density'] += 1
                continue

            # Check TOV solution
            try:
                solver = TOVSolver(eos)
                masses, radii = solver.mass_radius_relation()

                if len(masses) == 0:
                    constraint_violations['tov_failure'] += 1
                    continue

                mass_radius_data.append((masses, radii))
                valid_count += 1

            except Exception:
                constraint_violations['tov_failure'] += 1

        except Exception:
            # General failure
            continue

    fraction_valid = valid_count / n_check

    results = {
        'n_checked': n_check,
        'n_valid': valid_count,
        'fraction_valid': fraction_valid,
        'constraint_violations': constraint_violations,
        'mass_radius_data': mass_radius_data
    }

    if verbose:
        print("Physical constraint validation:")
        print(f"  Samples checked: {n_check}")
        print(f"  Valid samples: {valid_count} ({fraction_valid:.1%})")

        if constraint_violations['causality'] > 0:
            print(f"  Causality violations: {constraint_violations['causality']}")
        if constraint_violations['stability'] > 0:
            print(f"  Stability violations: {constraint_violations['stability']}")
        if constraint_violations['tov_failure'] > 0:
            print(f"  TOV failures: {constraint_violations['tov_failure']}")

    return results

def validate_observational_consistency(samples, observational_data, 
                                     eos_type='piecewise_polytrope',
                                     n_check=50, verbose=True):
    """
    Check consistency with observational constraints.

    Parameters:
    -----------
    samples : array_like
        Posterior samples
    observational_data : dict
        Observational constraints
    eos_type : str
        EoS parameterization type
    n_check : int
        Number of samples to check
    verbose : bool
        Print results

    Returns:
    --------
    dict : Observational consistency results
    """
    n_samples = len(samples)
    n_check = min(n_check, n_samples)

    check_indices = np.random.choice(n_samples, n_check, replace=False)
    check_samples = samples[check_indices]

    consistency_results = {
        'max_mass': [],
        'radius_14': [],
        'lambda_14': []
    }

    valid_mr_relations = []

    for params in check_samples:
        try:
            # Create EoS and solve TOV
            if eos_type == 'piecewise_polytrope':
                eos = PiecewisePolytrope(*params)
            else:
                continue

            solver = TOVSolver(eos)
            masses, radii = solver.mass_radius_relation()

            if len(masses) == 0:
                continue

            valid_mr_relations.append((masses, radii))

            # Maximum mass
            max_mass = np.max(masses)
            consistency_results['max_mass'].append(max_mass)

            # Radius at 1.4 solar masses
            if 1.4 >= np.min(masses) and 1.4 <= np.max(masses):
                r_14 = np.interp(1.4, masses, radii)
                consistency_results['radius_14'].append(r_14)

            # Tidal deformability at 1.4 solar masses
            try:
                lambda_14 = solver.calculate_tidal_deformability(1.4)
                if lambda_14 is not None and lambda_14 > 0:
                    consistency_results['lambda_14'].append(lambda_14)
            except:
                pass

        except Exception:
            continue

    # Check observational constraints
    obs_constraints = {}

    if len(consistency_results['max_mass']) > 0:
        max_masses = np.array(consistency_results['max_mass'])
        obs_constraints['max_mass'] = ObservationalConstraints.check_mass_constraint(
            (max_masses, np.zeros_like(max_masses))
        )

    if len(consistency_results['radius_14']) > 0:
        radii_14 = np.array(consistency_results['radius_14'])
        obs_constraints['radius'] = ObservationalConstraints.check_radius_constraint(
            (np.full_like(radii_14, 1.4), radii_14)
        )

    if len(consistency_results['lambda_14']) > 0:
        lambdas_14 = np.array(consistency_results['lambda_14'])
        obs_constraints['tidal'] = ObservationalConstraints.check_tidal_constraint(
            (np.full_like(lambdas_14, 1.4), lambdas_14)
        )

    results = {
        'n_checked': n_check,
        'consistency_data': consistency_results,
        'observational_constraints': obs_constraints,
        'valid_mr_relations': valid_mr_relations
    }

    if verbose:
        print("Observational consistency check:")
        print(f"  Samples checked: {n_check}")

        if 'max_mass' in obs_constraints:
            satisfied = obs_constraints['max_mass']['satisfied']
            max_mass_mean = np.mean(consistency_results['max_mass'])
            print(f"  Max mass constraint: {'✓' if satisfied else '✗'} (mean: {max_mass_mean:.2f} M☉)")

        if 'radius' in obs_constraints:
            satisfied = obs_constraints['radius']['satisfied']
            radius_mean = np.mean(consistency_results['radius_14'])
            print(f"  Radius constraint: {'✓' if satisfied else '✗'} (mean R₁.₄: {radius_mean:.1f} km)")

        if 'tidal' in obs_constraints:
            satisfied = obs_constraints['tidal']['satisfied']
            lambda_mean = np.mean(consistency_results['lambda_14'])
            print(f"  Tidal constraint: {'✓' if satisfied else '✗'} (mean Λ₁.₄: {lambda_mean:.0f})")

    return results

def statistical_tests(samples_1, samples_2, param_names=None):
    """
    Perform statistical tests comparing two sets of posterior samples.

    Parameters:
    -----------
    samples_1, samples_2 : array_like
        Two sets of posterior samples to compare
    param_names : list, optional
        Parameter names

    Returns:
    --------
    dict : Statistical test results
    """
    samples_1 = np.array(samples_1)
    samples_2 = np.array(samples_2)

    if param_names is None:
        param_names = [f'param_{i}' for i in range(samples_1.shape[1])]

    results = {}

    for i, param_name in enumerate(param_names):
        param_1 = samples_1[:, i]
        param_2 = samples_2[:, i]

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = kstest(param_1, param_2)

        # Mann-Whitney U test
        from scipy.stats import mannwhitneyu
        mw_stat, mw_p = mannwhitneyu(param_1, param_2, alternative='two-sided')

        # Wasserstein distance
        from scipy.stats import wasserstein_distance
        wd = wasserstein_distance(param_1, param_2)

        results[param_name] = {
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'mw_statistic': mw_stat,
            'mw_p_value': mw_p,
            'wasserstein_distance': wd
        }

    return results
