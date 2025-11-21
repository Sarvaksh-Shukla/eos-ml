"""
Forward models for neutron star observations.
Simulates observables from equation of state parameters through TOV solver.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from .tov import TOVSolver
from ..eos.eos_param import PiecewisePolytrope, SpectralEoS

class ObservationModel:
    """Base class for observation models."""

    def __init__(self, name, noise_level=0.1):
        self.name = name
        self.noise_level = noise_level

    def forward(self, eos_params):
        """Forward model: EoS parameters -> observables."""
        raise NotImplementedError

    def add_noise(self, true_observables):
        """Add observational noise to true values."""
        raise NotImplementedError

class PulsarTimingModel(ObservationModel):
    """Model for pulsar timing observations (masses)."""

    def __init__(self, pulsar_name, noise_level=0.05):
        super().__init__(f"Pulsar_{pulsar_name}", noise_level)
        self.pulsar_name = pulsar_name

    def forward(self, eos_params):
        """
        Forward model: EoS -> pulsar mass.

        This is simplified - in reality we'd need to know
        which specific neutron star we're observing.
        """
        # Create EoS from parameters
        if len(eos_params) == 4:  # Piecewise polytrope
            eos = PiecewisePolytrope(*eos_params)
        else:  # Spectral EoS
            eos = SpectralEoS(eos_params)

        # Solve TOV equations
        solver = TOVSolver(eos)

        try:
            masses, radii = solver.mass_radius_relation()
            if len(masses) == 0:
                return None

            # Return a "typical" mass (this is simplified)
            # In reality, we'd have specific information about each pulsar
            max_mass = np.max(masses)

            # Return mass somewhere in the range [1.2, max_mass]
            if max_mass > 1.2:
                typical_mass = np.random.uniform(1.2, min(max_mass, 2.5))
            else:
                return None

            return typical_mass

        except:
            return None

    def add_noise(self, true_mass):
        """Add Gaussian noise to mass measurement."""
        if true_mass is None:
            return None

        noise = np.random.normal(0, self.noise_level * true_mass)
        return true_mass + noise

class NICERModel(ObservationModel):
    """Model for NICER X-ray observations (mass + radius)."""

    def __init__(self, pulsar_name="J0030", noise_level=0.08):
        super().__init__(f"NICER_{pulsar_name}", noise_level)
        self.pulsar_name = pulsar_name

        # Correlation between mass and radius measurements
        self.correlation = -0.3  # Typical M-R anti-correlation

    def forward(self, eos_params):
        """Forward model: EoS -> (mass, radius) pair."""
        # Create EoS from parameters
        if len(eos_params) == 4:  # Piecewise polytrope
            eos = PiecewisePolytrope(*eos_params)
        else:  # Spectral EoS
            eos = SpectralEoS(eos_params)

        # Solve TOV equations
        solver = TOVSolver(eos)

        try:
            masses, radii = solver.mass_radius_relation()
            if len(masses) == 0:
                return None, None

            # Select a point on the M-R curve
            # For NICER, we typically observe lower-mass pulsars
            valid_indices = masses < 2.2  # Reasonable upper bound
            if not np.any(valid_indices):
                return None, None

            valid_masses = masses[valid_indices]
            valid_radii = radii[valid_indices]

            # Pick a random point from valid range
            if len(valid_masses) > 0:
                idx = np.random.randint(len(valid_masses))
                return valid_masses[idx], valid_radii[idx]
            else:
                return None, None

        except:
            return None, None

    def add_noise(self, true_values):
        """Add correlated noise to (mass, radius) measurements."""
        true_mass, true_radius = true_values

        if true_mass is None or true_radius is None:
            return None, None

        # Create covariance matrix
        sigma_m = self.noise_level * true_mass
        sigma_r = self.noise_level * true_radius

        cov_matrix = np.array([
            [sigma_m**2, self.correlation * sigma_m * sigma_r],
            [self.correlation * sigma_m * sigma_r, sigma_r**2]
        ])

        # Sample from multivariate normal
        noise = np.random.multivariate_normal([0, 0], cov_matrix)

        noisy_mass = true_mass + noise[0]
        noisy_radius = true_radius + noise[1]

        return noisy_mass, noisy_radius

class GravitationalWaveModel(ObservationModel):
    """Model for gravitational wave observations (tidal deformability)."""

    def __init__(self, event_name="GW170817", noise_level=0.3):
        super().__init__(f"GW_{event_name}", noise_level)
        self.event_name = event_name

        # Chirp mass and mass ratio (simplified)
        self.chirp_mass = 1.186  # Solar masses (GW170817)
        self.mass_ratio = 0.9    # Approximate

    def forward(self, eos_params):
        """Forward model: EoS -> tidal deformability."""
        # Create EoS from parameters
        if len(eos_params) == 4:  # Piecewise polytrope
            eos = PiecewisePolytrope(*eos_params)
        else:  # Spectral EoS
            eos = SpectralEoS(eos_params)

        # Solve TOV equations
        solver = TOVSolver(eos)

        try:
            # Calculate tidal deformability for both components
            m1, m2 = self._component_masses()

            lambda1 = solver.calculate_tidal_deformability(m1)
            lambda2 = solver.calculate_tidal_deformability(m2)

            if lambda1 is None or lambda2 is None:
                return None

            # Combined tidal parameter (simplified)
            lambda_tilde = (16.0/13.0) * ((m1 + 12*m2)*m1**4*lambda1 + 
                                         (m2 + 12*m1)*m2**4*lambda2) / (m1 + m2)**5

            return lambda_tilde

        except:
            return None

    def _component_masses(self):
        """Calculate component masses from chirp mass and mass ratio."""
        eta = self.mass_ratio / (1 + self.mass_ratio)**2
        total_mass = self.chirp_mass / (eta**(3/5))

        m1 = total_mass * self.mass_ratio / (1 + self.mass_ratio)
        m2 = total_mass / (1 + self.mass_ratio)

        return m1, m2

    def add_noise(self, true_lambda):
        """Add log-normal noise to tidal deformability."""
        if true_lambda is None or true_lambda <= 0:
            return None

        # Log-normal noise (tidal deformability is always positive)
        log_lambda = np.log(true_lambda)
        noisy_log_lambda = log_lambda + np.random.normal(0, self.noise_level)

        return np.exp(noisy_log_lambda)

class ForwardModelSuite:
    """Collection of forward models for multi-messenger observations."""

    def __init__(self, include_pulsar=True, include_nicer=True, include_gw=True):
        self.models = {}

        if include_pulsar:
            self.models['pulsar_timing'] = PulsarTimingModel("J0740")

        if include_nicer:
            self.models['nicer'] = NICERModel("J0030")

        if include_gw:
            self.models['gw'] = GravitationalWaveModel("GW170817")

    def simulate_observations(self, eos_params, add_noise=True):
        """
        Simulate all observations for given EoS parameters.

        Parameters:
        -----------
        eos_params : array_like
            EoS parameters
        add_noise : bool
            Whether to add observational noise

        Returns:
        --------
        dict : Simulated observations
        """
        observations = {}

        for model_name, model in self.models.items():
            # Generate true observable
            true_obs = model.forward(eos_params)

            if true_obs is None:
                observations[model_name] = None
                continue

            # Add noise if requested
            if add_noise:
                noisy_obs = model.add_noise(true_obs)
                observations[model_name] = {
                    'observed': noisy_obs,
                    'true': true_obs,
                    'model': model_name
                }
            else:
                observations[model_name] = {
                    'observed': true_obs,
                    'true': true_obs,
                    'model': model_name
                }

        return observations

    def generate_mock_dataset(self, n_samples, eos_sampler_func, add_noise=True):
        """
        Generate mock dataset of EoS parameters and observations.

        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        eos_sampler_func : callable
            Function that returns random EoS parameters
        add_noise : bool
            Whether to add observational noise

        Returns:
        --------
        tuple : (eos_parameters, observations)
        """
        eos_params_list = []
        observations_list = []

        valid_samples = 0
        attempts = 0
        max_attempts = n_samples * 10  # Avoid infinite loop

        while valid_samples < n_samples and attempts < max_attempts:
            attempts += 1

            # Sample EoS parameters
            eos_params = eos_sampler_func()

            # Simulate observations
            obs = self.simulate_observations(eos_params, add_noise)

            # Check if all observations are valid
            all_valid = True
            for model_name, model_obs in obs.items():
                if model_obs is None or model_obs['observed'] is None:
                    all_valid = False
                    break

            if all_valid:
                eos_params_list.append(eos_params)
                observations_list.append(obs)
                valid_samples += 1

        if valid_samples < n_samples:
            print(f"Warning: Only generated {valid_samples} valid samples out of {n_samples} requested")

        return np.array(eos_params_list), observations_list

def create_piecewise_polytrope_sampler(rng=None):
    """Create a sampler for piecewise polytrope parameters."""
    if rng is None:
        rng = np.random.default_rng()

    def sampler():
        log_p1 = rng.uniform(33.5, 35.5)
        gamma1 = rng.uniform(1.5, 4.5)
        gamma2 = rng.uniform(1.5, 4.5)
        gamma3 = rng.uniform(1.5, 4.5)
        return [log_p1, gamma1, gamma2, gamma3]

    return sampler

def create_spectral_eos_sampler(n_coeffs=4, rng=None):
    """Create a sampler for spectral EoS parameters."""
    if rng is None:
        rng = np.random.default_rng()

    def sampler():
        coeffs = [rng.normal(0.693, 0.1)]  # c_0
        for i in range(1, n_coeffs):
            coeffs.append(rng.normal(0, 0.5))
        return coeffs

    return sampler
