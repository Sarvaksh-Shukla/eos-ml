"""
MCMC sampling utilities for Bayesian inference of EoS parameters.
Implements various sampling algorithms and diagnostic tools.
"""

import numpy as np
import warnings
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import emcee

from ..physics.forward_model import ForwardModelSuite
from ..eos.eos_param import PiecewisePolytrope
from ..eos.constraints import EoSConstraints

class EoSSampler:
    """MCMC sampler for neutron star equation of state parameters."""

    def __init__(self, observation_data, eos_type='piecewise_polytrope', 
                 prior_bounds=None):
        """
        Initialize EoS parameter sampler.

        Parameters:
        -----------
        observation_data : dict
            Dictionary of observational data
        eos_type : str
            Type of EoS parameterization
        prior_bounds : list of tuples
            Prior bounds for each parameter
        """
        self.observation_data = observation_data
        self.eos_type = eos_type

        # Set up parameter space
        if eos_type == 'piecewise_polytrope':
            self.param_names = ['log_p1', 'gamma1', 'gamma2', 'gamma3']
            self.ndim = 4
            if prior_bounds is None:
                self.prior_bounds = [(33.0, 36.0), (1.0, 5.0), (1.0, 5.0), (1.0, 5.0)]
            else:
                self.prior_bounds = prior_bounds
        else:
            raise NotImplementedError(f"EoS type {eos_type} not implemented")

        # Initialize forward models
        self.forward_models = ForwardModelSuite()

        # Cache for likelihood calculations
        self._likelihood_cache = {}

    def log_prior(self, params):
        """Calculate log prior probability."""
        # Uniform priors within bounds
        for i, (param, (low, high)) in enumerate(zip(params, self.prior_bounds)):
            if not (low <= param <= high):
                return -np.inf

        # Additional physics-based constraints
        if self.eos_type == 'piecewise_polytrope':
            try:
                eos = PiecewisePolytrope(*params)
                constraint_results = EoSConstraints.check_all_constraints(eos)

                if not constraint_results['all_satisfied']:
                    return -np.inf

            except Exception:
                return -np.inf

        return 0.0  # Log of uniform prior normalization

    def log_likelihood(self, params):
        """Calculate log likelihood given parameters."""
        # Convert to tuple for hashing
        param_key = tuple(params)

        if param_key in self._likelihood_cache:
            return self._likelihood_cache[param_key]

        try:
            # Generate theoretical predictions
            theoretical_obs = self.forward_models.simulate_observations(
                params, add_noise=False
            )

            log_like = 0.0

            # Compare with each observation
            for obs_type, obs_data in self.observation_data.items():
                if obs_type not in theoretical_obs or theoretical_obs[obs_type] is None:
                    self._likelihood_cache[param_key] = -np.inf
                    return -np.inf

                theory_values = theoretical_obs[obs_type]['observed']

                if obs_type == 'pulsar_timing':
                    # Single mass measurement
                    observed_mass = obs_data['mass']
                    mass_error = obs_data.get('error', 0.1)

                    if theory_values is None:
                        self._likelihood_cache[param_key] = -np.inf
                        return -np.inf

                    log_like += -0.5 * ((theory_values - observed_mass) / mass_error)**2

                elif obs_type == 'nicer':
                    # Mass and radius measurement
                    observed_mass = obs_data['mass']
                    observed_radius = obs_data['radius']
                    mass_error = obs_data.get('mass_error', 0.1)
                    radius_error = obs_data.get('radius_error', 0.5)

                    if theory_values is None or len(theory_values) != 2:
                        self._likelihood_cache[param_key] = -np.inf
                        return -np.inf

                    theory_mass, theory_radius = theory_values

                    # Assume uncorrelated errors for simplicity
                    log_like += -0.5 * ((theory_mass - observed_mass) / mass_error)**2
                    log_like += -0.5 * ((theory_radius - observed_radius) / radius_error)**2

                elif obs_type == 'gw':
                    # Tidal deformability measurement
                    observed_lambda = obs_data['lambda']
                    lambda_error = obs_data.get('error', 50)

                    if theory_values is None or theory_values <= 0:
                        self._likelihood_cache[param_key] = -np.inf
                        return -np.inf

                    # Log-normal likelihood for Lambda (always positive)
                    log_like += -0.5 * ((np.log(theory_values) - np.log(observed_lambda)) / 
                                       (lambda_error / observed_lambda))**2

            self._likelihood_cache[param_key] = log_like
            return log_like

        except Exception as e:
            self._likelihood_cache[param_key] = -np.inf
            return -np.inf

    def log_posterior(self, params):
        """Calculate log posterior probability."""
        log_prior_val = self.log_prior(params)
        if log_prior_val == -np.inf:
            return -np.inf

        log_like_val = self.log_likelihood(params)
        return log_prior_val + log_like_val

    def find_map_estimate(self, n_starts=10):
        """Find maximum a posteriori (MAP) estimate."""
        best_result = None
        best_log_prob = -np.inf

        for i in range(n_starts):
            # Random starting point
            start_params = []
            for low, high in self.prior_bounds:
                start_params.append(np.random.uniform(low, high))

            # Minimize negative log posterior
            result = minimize(
                lambda x: -self.log_posterior(x),
                start_params,
                method='L-BFGS-B',
                bounds=self.prior_bounds
            )

            if result.success and -result.fun > best_log_prob:
                best_result = result
                best_log_prob = -result.fun

        if best_result is not None:
            return best_result.x, best_log_prob
        else:
            return None, -np.inf

    def run_mcmc(self, n_walkers=32, n_steps=5000, n_burn=1000, 
                 start_params=None, progress=True):
        """
        Run MCMC sampling using emcee.

        Parameters:
        -----------
        n_walkers : int
            Number of MCMC walkers
        n_steps : int
            Number of MCMC steps
        n_burn : int
            Number of burn-in steps to discard
        start_params : array_like, optional
            Starting parameters (will find MAP if not provided)
        progress : bool
            Show progress bar

        Returns:
        --------
        tuple : (samples, log_probs, sampler)
        """
        # Initialize starting positions
        if start_params is None:
            map_params, _ = self.find_map_estimate()
            if map_params is None:
                # Use random starting points
                start_params = []
                for low, high in self.prior_bounds:
                    start_params.append((low + high) / 2)

        # Initialize walkers around starting point
        pos = []
        for i in range(n_walkers):
            walker_pos = []
            for j, (param, (low, high)) in enumerate(zip(start_params, self.prior_bounds)):
                # Small Gaussian perturbation
                sigma = (high - low) * 0.01
                new_param = np.random.normal(param, sigma)
                new_param = np.clip(new_param, low, high)
                walker_pos.append(new_param)
            pos.append(walker_pos)

        # Set up sampler
        sampler = emcee.EnsembleSampler(
            n_walkers, self.ndim, self.log_posterior
        )

        # Run MCMC
        if progress:
            from tqdm import tqdm
            with tqdm(total=n_steps, desc="MCMC sampling") as pbar:
                for i, state in enumerate(sampler.sample(pos, iterations=n_steps)):
                    pbar.update(1)
        else:
            sampler.run_mcmc(pos, n_steps)

        # Extract samples (discard burn-in)
        samples = sampler.get_chain(discard=n_burn, flat=True)
        log_probs = sampler.get_log_prob(discard=n_burn, flat=True)

        return samples, log_probs, sampler

    def calculate_evidence(self, samples, log_probs):
        """Calculate Bayesian evidence using thermodynamic integration."""
        # This is a simplified implementation
        # For accurate evidence calculation, use nested sampling (e.g., dynesty)

        # Harmonic mean estimator (can be unstable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_evidence = -np.log(np.mean(np.exp(-log_probs)))

        return log_evidence

    def diagnostic_plots(self, samples, sampler=None):
        """Generate diagnostic plots for MCMC convergence."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(self.ndim, 2, figsize=(12, 2*self.ndim))

        for i in range(self.ndim):
            # Trace plot
            if sampler is not None:
                chain = sampler.get_chain()[:, :, i]
                for walker in range(chain.shape[1]):
                    axes[i, 0].plot(chain[:, walker], alpha=0.3)
            axes[i, 0].set_xlabel('Step')
            axes[i, 0].set_ylabel(self.param_names[i])
            axes[i, 0].set_title(f'Trace: {self.param_names[i]}')

            # Histogram
            axes[i, 1].hist(samples[:, i], bins=50, alpha=0.7)
            axes[i, 1].set_xlabel(self.param_names[i])
            axes[i, 1].set_ylabel('Density')
            axes[i, 1].set_title(f'Posterior: {self.param_names[i]}')

        plt.tight_layout()
        return fig

class MultiModalSampler:
    """Advanced sampler for multi-modal posterior distributions."""

    def __init__(self, observation_data, eos_type='piecewise_polytrope'):
        self.base_sampler = EoSSampler(observation_data, eos_type)

    def run_nested_sampling(self, nlive=500, dlogz=0.1):
        """
        Run nested sampling for multi-modal distributions.
        Requires dynesty package.
        """
        try:
            import dynesty
        except ImportError:
            raise ImportError("dynesty package required for nested sampling")

        # Prior transform function
        def prior_transform(u):
            params = []
            for i, (low, high) in enumerate(self.base_sampler.prior_bounds):
                params.append(low + u[i] * (high - low))
            return np.array(params)

        # Set up nested sampler
        sampler = dynesty.NestedSampler(
            self.base_sampler.log_likelihood,
            prior_transform,
            self.base_sampler.ndim,
            nlive=nlive
        )

        # Run sampling
        sampler.run_nested(dlogz=dlogz)

        # Extract results
        results = sampler.results
        samples = results.samples
        weights = np.exp(results.logwt - results.logz[-1])

        return samples, weights, results.logz[-1]  # samples, weights, log_evidence
