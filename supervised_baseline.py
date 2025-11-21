"""
Supervised learning baseline for EoS parameter inference.
Neural network model for direct parameter estimation from observations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path

class EoSDataset(Dataset):
    """PyTorch dataset for EoS parameter inference."""

    def __init__(self, observations, parameters):
        """
        Initialize dataset.

        Parameters:
        -----------
        observations : array_like
            Observational data (n_samples, n_obs_features)
        parameters : array_like
            EoS parameters (n_samples, n_params)
        """
        self.observations = torch.FloatTensor(observations)
        self.parameters = torch.FloatTensor(parameters)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.parameters[idx]

class SupervisedEoSNet(nn.Module):
    """Neural network for EoS parameter regression."""

    def __init__(self, n_obs_features=4, n_params=4, hidden_dims=[128, 256, 128]):
        """
        Initialize neural network.

        Parameters:
        -----------
        n_obs_features : int
            Number of observational features
        n_params : int
            Number of EoS parameters to predict
        hidden_dims : list
            Hidden layer dimensions
        """
        super(SupervisedEoSNet, self).__init__()

        self.n_obs_features = n_obs_features
        self.n_params = n_params

        # Build network layers
        layers = []
        prev_dim = n_obs_features

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, n_params))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        """Forward pass through network."""
        return self.network(x)

class SupervisedBaseline:
    """Supervised learning baseline for EoS inference."""

    def __init__(self, n_obs_features=4, n_params=4, hidden_dims=[128, 256, 128]):
        """
        Initialize supervised baseline.

        Parameters:
        -----------
        n_obs_features : int
            Number of observational features
        n_params : int
            Number of EoS parameters
        hidden_dims : list
            Hidden layer dimensions
        """
        self.n_obs_features = n_obs_features
        self.n_params = n_params
        self.hidden_dims = hidden_dims

        # Initialize model
        self.model = SupervisedEoSNet(n_obs_features, n_params, hidden_dims)
        self.normalization_info = None
        self.training_history = {'train_loss': [], 'val_loss': []}

        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_data(self, data_path):
        """Load training data from file."""
        from .dataset import DatasetLoader

        loader = DatasetLoader(data_path)
        parameters, observations = loader.load_synthetic_dataset()

        # Normalize data
        norm_params, norm_obs, norm_info = loader.normalize_data(parameters, observations)
        self.normalization_info = norm_info

        # Split into train/validation
        (train_params, train_obs), (val_params, val_obs) = loader.get_train_test_split()

        # Normalize split data
        train_params_norm = (train_params - norm_info['param_min']) / (norm_info['param_max'] - norm_info['param_min'])
        val_params_norm = (val_params - norm_info['param_min']) / (norm_info['param_max'] - norm_info['param_min'])
        train_obs_norm = (train_obs - norm_info['obs_mean']) / norm_info['obs_std']
        val_obs_norm = (val_obs - norm_info['obs_mean']) / norm_info['obs_std']

        # Create datasets
        self.train_dataset = EoSDataset(train_obs_norm, train_params_norm)
        self.val_dataset = EoSDataset(val_obs_norm, val_params_norm)

        print(f"Loaded training data: {len(self.train_dataset)} train, {len(self.val_dataset)} val")

    def train(self, epochs=100, batch_size=32, lr=1e-3, weight_decay=1e-5):
        """
        Train the supervised baseline model.

        Parameters:
        -----------
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        lr : float
            Learning rate
        weight_decay : float
            Weight decay for regularization
        """
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )

        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Training loop
        self.model.train()
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            for batch_obs, batch_params in train_loader:
                batch_obs = batch_obs.to(self.device)
                batch_params = batch_params.to(self.device)

                optimizer.zero_grad()
                pred_params = self.model(batch_obs)
                loss = criterion(pred_params, batch_params)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_obs, batch_params in val_loader:
                    batch_obs = batch_obs.to(self.device)
                    batch_params = batch_params.to(self.device)

                    pred_params = self.model(batch_obs)
                    loss = criterion(pred_params, batch_params)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self.model.train()

            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

        # Load best model
        self.model.load_state_dict(self.best_model_state)
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

    def predict(self, observations, return_uncertainty=False):
        """
        Make predictions on new observations.

        Parameters:
        -----------
        observations : array_like
            Observational data to predict on
        return_uncertainty : bool
            Whether to return prediction uncertainty (using dropout)

        Returns:
        --------
        array_like : Predicted EoS parameters
        """
        # Normalize observations
        if self.normalization_info is None:
            raise ValueError("Model must be trained before making predictions")

        obs_normalized = (observations - self.normalization_info['obs_mean']) / self.normalization_info['obs_std']
        obs_tensor = torch.FloatTensor(obs_normalized).to(self.device)

        self.model.eval()

        if return_uncertainty:
            # Monte Carlo dropout for uncertainty estimation
            n_samples = 100
            predictions = []

            # Enable dropout during inference
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()

            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.model(obs_tensor)
                    predictions.append(pred.cpu().numpy())

            predictions = np.array(predictions)

            # Denormalize predictions
            param_min = self.normalization_info['param_min']
            param_max = self.normalization_info['param_max']

            predictions_denorm = predictions * (param_max - param_min) + param_min

            mean_pred = np.mean(predictions_denorm, axis=0)
            std_pred = np.std(predictions_denorm, axis=0)

            return mean_pred, std_pred

        else:
            with torch.no_grad():
                pred_normalized = self.model(obs_tensor).cpu().numpy()

            # Denormalize predictions
            param_min = self.normalization_info['param_min']
            param_max = self.normalization_info['param_max']

            predictions = pred_normalized * (param_max - param_min) + param_min

            return predictions

    def evaluate(self, test_observations, test_parameters):
        """Evaluate model on test data."""
        predictions = self.predict(test_observations)

        # Calculate metrics
        mse = np.mean((predictions - test_parameters)**2, axis=0)
        mae = np.mean(np.abs(predictions - test_parameters), axis=0)

        # R-squared for each parameter
        r2 = []
        for i in range(self.n_params):
            ss_res = np.sum((test_parameters[:, i] - predictions[:, i])**2)
            ss_tot = np.sum((test_parameters[:, i] - np.mean(test_parameters[:, i]))**2)
            r2.append(1 - (ss_res / ss_tot))

        results = {
            'mse': mse,
            'mae': mae,
            'r2': np.array(r2),
            'overall_mse': np.mean(mse),
            'overall_mae': np.mean(mae),
            'overall_r2': np.mean(r2)
        }

        return results

    def save(self, save_path):
        """Save trained model to file."""
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'normalization_info': self.normalization_info,
            'training_history': self.training_history,
            'model_config': {
                'n_obs_features': self.n_obs_features,
                'n_params': self.n_params,
                'hidden_dims': self.hidden_dims
            }
        }

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_data, save_path)
        print(f"Model saved to {save_path}")

    def load(self, save_path):
        """Load trained model from file."""
        save_data = torch.load(save_path, map_location=self.device)

        # Reconstruct model if needed
        config = save_data['model_config']
        if (config['n_obs_features'] != self.n_obs_features or 
            config['n_params'] != self.n_params or
            config['hidden_dims'] != self.hidden_dims):

            self.n_obs_features = config['n_obs_features']
            self.n_params = config['n_params']
            self.hidden_dims = config['hidden_dims']

            self.model = SupervisedEoSNet(
                self.n_obs_features, self.n_params, self.hidden_dims
            ).to(self.device)

        # Load state
        self.model.load_state_dict(save_data['model_state_dict'])
        self.normalization_info = save_data['normalization_info']
        self.training_history = save_data['training_history']

        print(f"Model loaded from {save_path}")

    def load_observational_data(self, obs_data_path):
        """Load observational data for inference."""
        from .dataset import ObservationalDataLoader

        loader = ObservationalDataLoader()
        obs_data = loader.load_all_observational_data()

        # Convert to format expected by model
        # This is simplified - in practice you'd need to handle different data formats

        formatted_obs = []

        if obs_data['nicer'] is not None:
            # Use mean values from NICER data
            nicer_data = obs_data['nicer']
            mean_mass = nicer_data['mass'].mean()
            mean_radius = nicer_data['radius'].mean()
            formatted_obs.extend([mean_mass, mean_radius])
        else:
            formatted_obs.extend([1.4, 12.0])  # Default values

        if obs_data['gw'] is not None:
            # Use mean tidal deformability
            gw_data = obs_data['gw']
            mean_lambda = gw_data['lambda_tilde'].mean()
            formatted_obs.append(mean_lambda)
        else:
            formatted_obs.append(190.0)  # Default value

        # Add pulsar timing (placeholder)
        formatted_obs.insert(0, 2.0)  # Heavy pulsar mass

        return np.array(formatted_obs).reshape(1, -1)

    def infer(self, obs_data):
        """Run inference on observational data."""
        predictions, uncertainties = self.predict(obs_data, return_uncertainty=True)

        # Create samples for compatibility with other methods
        n_samples = 1000
        samples = []

        for i in range(n_samples):
            sample = np.random.normal(predictions[0], uncertainties[0])
            samples.append(sample)

        return np.array(samples)

    def plot_training_history(self):
        """Plot training and validation loss curves."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        ax.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True)

        return fig

