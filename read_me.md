# EoS-ML: End-to-End Neutron Star Equation of State Inference

A practical, minimal-but-complete pipeline to infer and verify the neutron-star equation of state (EoS) with machine learning.

## Overview

This project implements a complete workflow for constraining neutron star equations of state using multi-messenger astronomical observations. It combines:

- **Physical Models**: TOV equation solver, EoS parameterizations, forward models
- **Machine Learning**: Supervised baselines and simulation-based inference (SBI)  
- **Observational Data**: NICER X-ray timing, gravitational waves, pulsar timing
- **Validation Tools**: Physical constraints, statistical checks, visualization

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd eos-ml

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

1. **Generate synthetic training data:**
```bash
python main.py generate --n-samples 10000 --output-path data/synthetic_dataset.npz
```

2. **Train a supervised baseline:**
```bash
python main.py train-baseline --data-path data/synthetic_dataset.npz --epochs 100
```

3. **Train SBI model:**
```bash
python main.py train-sbi --data-path data/synthetic_dataset.npz --num-simulations 50000
```

4. **Run inference on observational data:**
```bash
python main.py infer --method sbi --model-path models/sbi_flows.pkl --obs-data-path data/
```

### Interactive Usage

```python
from inference.dataset import SyntheticDatasetGenerator
from inference.sbi_flows import SBIFlows
from verify.plots import create_corner_plot

# Generate training data
generator = SyntheticDatasetGenerator(n_samples=5000)
generator.generate_and_save('data/training.npz')

# Train SBI model
sbi_model = SBIFlows()
sbi_model.load_data('data/training.npz')
sbi_model.train(num_simulations=20000)

# Load observational data and run inference
obs_data = sbi_model.load_observational_data('data/')
posterior_samples = sbi_model.infer(obs_data, num_samples=1000)

# Create plots
fig = create_corner_plot(posterior_samples)
```

## Project Structure

```
eos-ml/
├── requirements.txt          # Python dependencies
├── main.py                   # Main CLI interface
├── data/                     # Data files and configuration
│   ├── nicer_samples.csv     # NICER observational data
│   ├── gw_tidal_samples.csv  # Gravitational wave constraints
│   ├── priors_config.json    # Prior configuration
│   └── raw_events/           # Raw observational files
├── eos/                      # Equation of state models
│   ├── units.py             # Physical units and constants
│   ├── eos_param.py         # EoS parameterizations
│   └── constraints.py       # Physical constraints
├── physics/                  # Physical models
│   ├── tov.py               # TOV equation solver
│   └── forward_model.py     # Forward models for observations
├── inference/                # Machine learning inference
│   ├── dataset.py           # Data generation and loading
│   ├── sampler.py           # MCMC sampling utilities
│   ├── supervised_baseline.py # Supervised learning baseline
│   └── sbi_flows.py         # Simulation-based inference
└── verify/                   # Validation and visualization
    ├── checks.py            # Validation utilities
    └── plots.py             # Plotting functions
```

## Key Features

### Physical Models
- **TOV Solver**: Solves Tolman-Oppenheimer-Volkoff equations for neutron star structure
- **EoS Parameterizations**: Piecewise polytropes, spectral decomposition
- **Constraints**: Causality, thermodynamic stability, observational bounds

### Machine Learning Methods
- **Supervised Baseline**: Neural network for direct parameter estimation
- **Simulation-Based Inference**: Neural posterior estimation with normalizing flows
- **Ensemble Methods**: Multiple model averaging for robustness

### Multi-Messenger Observations
- **NICER**: X-ray timing constraints on mass and radius
- **Gravitational Waves**: Tidal deformability from binary mergers  
- **Pulsar Timing**: Precise mass measurements of neutron stars

### Validation Framework
- **Physical Validation**: Check causality, stability, TOV solutions
- **Statistical Tests**: Convergence diagnostics, effective sample size
- **Posterior Predictive Checks**: Model adequacy assessment

## Scientific Background

Neutron stars are the densest objects in the observable universe, with central densities exceeding nuclear saturation density. The equation of state of matter at these extreme densities is poorly constrained by laboratory experiments, making neutron star observations crucial for understanding fundamental physics.

This pipeline implements state-of-the-art methods for combining multi-messenger observations to constrain the neutron star EoS:

1. **Forward Modeling**: Physical models (TOV equations) connect EoS parameters to observables
2. **Bayesian Inference**: Posterior sampling accounts for observational uncertainties  
3. **Machine Learning**: SBI enables likelihood-free inference for complex forward models
4. **Multi-Messenger**: Combines X-ray, gravitational wave, and timing observations

## Dependencies

Core scientific computing:
- numpy, scipy, pandas, matplotlib, seaborn

Machine learning:
- torch, scikit-learn

Simulation-based inference:
- sbi, pyro-ppl

Astronomy/physics:
- astropy

MCMC sampling:
- emcee

Optional for advanced features:
- corner (corner plots)
- dynesty (nested sampling)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{eos_ml,
  title={EoS-ML: End-to-End Neutron Star Equation of State Inference},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for any improvements.

## Acknowledgments

This work builds on decades of research in neutron star physics, gravitational wave astronomy, and Bayesian inference. Special thanks to the LIGO-Virgo-KAGRA collaboration, NICER team, and the broader neutron star community.
