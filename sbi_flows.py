"""
Simulation-based inference using normalizing flows for EoS parameter estimation.
Implements neural posterior estimation (NPE) for likelihood-free inference.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pickle

try:
    from sbi import utils as sbi_utils
    from sbi import analysis as sbi_analysis
    from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
    SBI_AVAILABLE = True
except ImportError:
    SBI_AVAILABLE = False
    print("Warning: sbi package not available. SBI functionality will be limited.")

from ..physics.forward_model import ForwardModelSuite, create_piecewise_polytrope_sampler
from ..eos.eos_param import PiecewisePolytrope

class SBIFlows:
    """Simulation-based inference using normalizing flows."""

    def __init__(self, eos_type='piecewise_polytrope', prior_bounds=None):
        """
        Initialize SBI flow model.

        Parameters:
        -----------
        eos_type : str
            Type of EoS parameterization
        prior_bounds : list of tuples
            Prior bounds for parameters
        """
        if not SBI_AVAILABLE:
            raise ImportError("sbi package required for SBI functionality. Install with: pip install sbi")

        self.eos_type = eos_type

        # Set up parameter space
        if eos_type == 'piecewise_polytrope':
            self.param_names = ['log_p1', 'gamma1', 'gamma2', 'gamma3']
            self.n_params = 4
            if prior_bounds is None:
                self.prior_bounds = [(33.0, 36.0), (1.0, 5.0), (1.0, 5.0), (1.0, 5.0)]
            else:
                self.prior_bounds = prior_bounds
        else:
            raise NotImplementedError(f"EoS type {eos_type} not implemented")

        # Set up prior distribution
        prior_min = torch.tensor([bound[0] for bound in self.prior_bounds])
        prior_max = torch.tensor([bound[1] for bound in self.prior_bounds])
        self.prior = sbi_utils.BoxUniform(low=prior_min, high=prior_max)

        # Initialize forward models
        self.forward_models = ForwardModelSuite()

        # SBI components
        self.posterior = None
        self.inference = None
        self.simulator = None

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _simulator_function(self, params):
        """
        Simulator function for SBI.
        Takes EoS parameters and returns observables.
        """
        # Convert torch tensor to numpy if needed
        if torch.is_tensor(params):
            params_np = params.numpy()
        else:
            params_np = np.array(params)

        # Handle batch dimension
        if params_np.ndim == 1:
            params_np = params_np.reshape(1, -1)

        results = []

        for param_row in params_np:
            try:
                # Simulate observations
                observations = self.forward_models.simulate_observations(
                    param_row, add_noise=True
                )

                # Extract observables in consistent order
                obs_vector = []

                # Pulsar timing (mass)
                if 'pulsar_timing' in observations and observations['pulsar_timing'] is not None:
                    obs_vector.append(observations['pulsar_timing']['observed'])
                else:
                    obs_vector.append(np.nan)

                # NICER (mass, radius)
                if 'nicer' in observations and observations['nicer'] is not None:
                    nicer_obs = observations['nicer']['observed']
                    if nicer_obs is not None and len(nicer_obs) == 2:
                        obs_vector.extend(nicer_obs)
                    else:
                        obs_vector.extend([np.nan, np.nan])
                else:
                    obs_vector.extend([np.nan, np.nan])

                # Gravitational waves (tidal deformability)
                if 'gw' in observations and observations['gw'] is not None:
                    obs_vector.append(observations['gw']['observed'])
                else:
                    obs_vector.append(np.nan)

                # Check for any NaN values
                if np.any(np.isnan(obs_vector)):
                    # Return default values that won't crash the inference
                    obs_vector = [2.0, 1.4, 12.0, 190.0]

                results.append(obs_vector)

            except Exception as e:
                # Return default values if simulation fails
                results.append([2.0, 1.4, 12.0, 190.0])

        result_tensor = torch.tensor(results, dtype=torch.float32)

        # Return single row if input was single parameter set
        if result_tensor.shape[0] == 1:
            return result_tensor.squeeze(0)
        return result_tensor

    def load_data(self, data_path):
        """Load synthetic dataset for training."""
        from .dataset import DatasetLoader

        loader = DatasetLoader(data_path)
        parameters, observations = loader.load_synthetic_dataset()

        # Convert to torch tensors
        self.theta_train = torch.tensor(parameters, dtype=torch.float32)
        self.x_train = torch.tensor(observations, dtype=torch.float32)

        print(f"Loaded SBI training data: {len(self.theta_train)} samples")

    def train(self, num_simulations=10000, num_rounds=1, neural_net='maf'):
        """
        Train the SBI model using neural posterior estimation.

        Parameters:
        -----------
        num_simulations : int
            Number of simulations to run
        num_rounds : int
            Number of inference rounds
        neural_net : str
            Type of neural network ('maf', 'nsf', 'mdn')
        """
        print(f"Training SBI model with {num_simulations} simulations...")

        # Set up simulator and prior for SBI
        simulator, prior = prepare_for_sbi(self._simulator_function, self.prior)

        # Initialize inference method
        self.inference = SNPE(prior=prior, density_estimator=neural_net, device=self.device)

        posteriors = []
        proposal = prior

        for round_idx in range(num_rounds):
            print(f"Round {round_idx + 1}/{num_rounds}")

            # Run simulations
            theta, x = simulate_for_sbi(
                simulator, proposal, num_simulations=num_simulations,
                simulation_batch_size=100
            )

            # Append to training data if we have pre-loaded data
            if hasattr(self, 'theta_train') and hasattr(self, 'x_train'):
                theta = torch.cat([theta, self.theta_train], dim=0)
                x = torch.cat([x, self.x_train], dim=0)

            # Train posterior estimator
            density_estimator = self.inference.append_simulations(theta, x).train()
            posterior = self.inference.build_posterior(density_estimator)

            posteriors.append(posterior)
            proposal = posterior.set_default_x(x[-1000:].mean(dim=0))  # Use recent simulations

        self.posterior = posteriors[-1]  # Use final posterior
        print("SBI training completed!")

    def infer(self, observation, num_samples=1000):
        """
        Perform inference on observational data.

        Parameters:
        -----------
        observation : array_like
            Observational data
        num_samples : int
            Number of posterior samples to generate

        Returns:
        --------
        array_like : Posterior samples
        """
        if self.posterior is None:
            raise ValueError("Model must be trained before inference")

        # Convert observation to torch tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        # Sample from posterior
        samples = self.posterior.sample((num_samples,), x=obs_tensor[0])

        return samples.numpy()

    def save(self, save_path):
        """Save trained SBI model."""
        if self.posterior is None:
            raise ValueError("No trained model to save")

        save_data = {
            'posterior': self.posterior,
            'inference': self.inference,
            'prior_bounds': self.prior_bounds,
            'param_names': self.param_names,
            'eos_type': self.eos_type
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"SBI model saved to {save_path}")

    def load(self, save_path):
        """Load trained SBI model."""
        with open(save_path, 'rb') as f:
            save_data = pickle.load(f)

        self.posterior = save_data['posterior']
        self.inference = save_data['inference']
        self.prior_bounds = save_data['prior_bounds']
        self.param_names = save_data['param_names']
        self.eos_type = save_data['eos_type']

        print(f"SBI model loaded from {save_path}")

    def load_observational_data(self, obs_data_path):
        """Load observational data for inference."""
        from .dataset import ObservationalDataLoader

        loader = ObservationalDataLoader()
        obs_data = loader.load_all_observational_data()

        # Convert to format expected by SBI model
        formatted_obs = []

        # Pulsar timing mass (using heavy pulsar as example)
        formatted_obs.append(2.08)  # J0740+6620 mass

        if obs_data['nicer'] is not None:
            nicer_data = obs_data['nicer']
            mean_mass = nicer_data['mass'].mean()
            mean_radius = nicer_data['radius'].mean()
            formatted_obs.extend([mean_mass, mean_radius])
        else:
            formatted_obs.extend([1.4, 12.0])

        if obs_data['gw'] is not None:
            gw_data = obs_data['gw']
            mean_lambda = gw_data['lambda_tilde'].mean()
            formatted_obs.append(mean_lambda)
        else:
            formatted_obs.append(190.0)

        return np.array(formatted_obs)

    def posterior_predictive_check(self, samples, observation, n_checks=100):
        """
        Perform posterior predictive checks.

        Parameters:
        -----------
        samples : array_like
            Posterior samples
        observation : array_like
            Original observation
        n_checks : int
            Number of predictive samples

        Returns:
        --------
        dict : Posterior predictive check results
        """
        # Sample subset of posterior samples
        n_samples = min(n_checks, len(samples))
        sample_indices = np.random.choice(len(samples), n_samples, replace=False)
        selected_samples = samples[sample_indices]

        # Generate predictions from selected samples
        predictions = []
        for sample in selected_samples:
            pred = self._simulator_function(sample)
            if torch.is_tensor(pred):
                pred = pred.numpy()
            predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate statistics
        pred_mean = np.mean(predictions, axis=0)
        pred_std = np.std(predictions, axis=0)

        # Compare with observation
        obs_array = np.array(observation).flatten()

        # Calculate p-values (two-tailed test)
        p_values = []
        for i in range(len(obs_array)):
            if pred_std[i] > 0:
                z_score = (obs_array[i] - pred_mean[i]) / pred_std[i]
                p_val = 2 * (1 - abs(z_score))  # Simplified p-value
                p_values.append(max(0, min(1, p_val)))
            else:
                p_values.append(1.0)

        return {
            'predicted_mean': pred_mean,
            'predicted_std': pred_std,
            'observed': obs_array,
            'p_values': np.array(p_values),
            'predictions': predictions
        }

    def plot_posterior(self, samples, save_path=None):
        """Plot posterior distributions."""
        if not SBI_AVAILABLE:
            print("SBI plotting requires the sbi package")
            return None

        try:
            import matplotlib.pyplot as plt

            # Convert to torch tensor if needed
            if isinstance(samples, np.ndarray):
                samples_tensor = torch.tensor(samples)
            else:
                samples_tensor = samples

            # Create corner plot using SBI analysis tools
            fig, axes = sbi_analysis.pairplot(
                samples_tensor,
                labels=self.param_names,
                figsize=(8, 8)
            )

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Posterior plot saved to {save_path}")

            return fig

        except Exception as e:
            print(f"Could not create posterior plot: {e}")
            return None

class EnsembleSBI:
    """Ensemble of SBI models for improved robustness."""

    def __init__(self, n_models=3, **kwargs):
        """
        Initialize ensemble of SBI models.

        Parameters:
        -----------
        n_models : int
            Number of models in ensemble
        **kwargs : dict
            Arguments passed to individual SBI models
        """
        self.n_models = n_models
        self.models = [SBIFlows(**kwargs) for _ in range(n_models)]
        self.trained = False

    def load_data(self, data_path):
        """Load data for all models in ensemble."""
        for model in self.models:
            model.load_data(data_path)

    def train(self, num_simulations=10000, **kwargs):
        """Train all models in ensemble."""
        print(f"Training ensemble of {self.n_models} SBI models...")

        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_models}")

            # Use different random seeds for diversity
            np.random.seed(42 + i)
            torch.manual_seed(42 + i)

            model.train(num_simulations=num_simulations, **kwargs)

        self.trained = True
        print("Ensemble training completed!")

    def infer(self, observation, num_samples=1000):
        """Perform ensemble inference."""
        if not self.trained:
            raise ValueError("Ensemble must be trained before inference")

        all_samples = []

        for model in self.models:
            samples = model.infer(observation, num_samples // self.n_models)
            all_samples.append(samples)

        # Combine samples from all models
        combined_samples = np.vstack(all_samples)

        return combined_samples

    def save(self, save_dir):
        """Save all models in ensemble."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, model in enumerate(self.models):
            model_path = save_dir / f"sbi_model_{i}.pkl"
            model.save(model_path)

        # Save ensemble metadata
        metadata = {
            'n_models': self.n_models,
            'trained': self.trained
        }

        with open(save_dir / 'ensemble_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Ensemble saved to {save_dir}")

    def load(self, save_dir):
        """Load all models in ensemble."""
        save_dir = Path(save_dir)

        # Load metadata
        with open(save_dir / 'ensemble_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        self.n_models = metadata['n_models']
        self.trained = metadata['trained']

        # Load individual models
        self.models = []
        for i in range(self.n_models):
            model = SBIFlows()
            model_path = save_dir / f"sbi_model_{i}.pkl"
            model.load(model_path)
            self.models.append(model)

        print(f"Ensemble loaded from {save_dir}")
