#!/usr/bin/env python3
"""
EoS-ML: End-to-End Neutron Star Equation of State Inference Pipeline

Main entry point for training models, generating synthetic data, 
and running inference on neutron star equation of state parameters.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from inference.dataset import SyntheticDatasetGenerator
from inference.supervised_baseline import SupervisedBaseline
from inference.sbi_flows import SBIFlows
from verify.plots import create_corner_plot, create_mass_radius_plot
from verify.checks import validate_eos_parameters

def generate_synthetic_data(args):
    """Generate synthetic training dataset."""
    print("Generating synthetic dataset...")
    generator = SyntheticDatasetGenerator(
        n_samples=args.n_samples,
        noise_level=args.noise_level
    )
    generator.generate_and_save(args.output_path)
    print(f"Synthetic dataset saved to {args.output_path}")

def train_baseline(args):
    """Train supervised baseline model."""
    print("Training supervised baseline...")
    model = SupervisedBaseline()
    model.load_data(args.data_path)
    model.train(epochs=args.epochs, lr=args.learning_rate)
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

def train_sbi(args):
    """Train simulation-based inference model."""
    print("Training SBI model...")
    sbi_model = SBIFlows()
    sbi_model.load_data(args.data_path)
    sbi_model.train(num_simulations=args.num_simulations)
    sbi_model.save(args.model_path)
    print(f"SBI model saved to {args.model_path}")

def run_inference(args):
    """Run inference on observational data."""
    print(f"Running inference using {args.method}...")

    if args.method == 'baseline':
        model = SupervisedBaseline()
        model.load(args.model_path)
    elif args.method == 'sbi':
        model = SBIFlows()
        model.load(args.model_path)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Load observational data
    obs_data = model.load_observational_data(args.obs_data_path)

    # Run inference
    posterior_samples = model.infer(obs_data)

    # Validate results
    validate_eos_parameters(posterior_samples)

    # Create plots
    create_corner_plot(posterior_samples, save_path=args.output_dir)
    create_mass_radius_plot(posterior_samples, save_path=args.output_dir)

    print(f"Inference results saved to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='EoS-ML: Neutron Star EoS Inference Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate synthetic data
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic training data')
    gen_parser.add_argument('--n-samples', type=int, default=10000, 
                           help='Number of synthetic samples to generate')
    gen_parser.add_argument('--noise-level', type=float, default=0.1,
                           help='Noise level for synthetic observations')
    gen_parser.add_argument('--output-path', type=str, default='data/synthetic_dataset.npz',
                           help='Output path for synthetic dataset')

    # Train baseline model
    baseline_parser = subparsers.add_parser('train-baseline', help='Train supervised baseline')
    baseline_parser.add_argument('--data-path', type=str, default='data/synthetic_dataset.npz',
                                help='Path to training data')
    baseline_parser.add_argument('--epochs', type=int, default=100,
                                help='Number of training epochs')
    baseline_parser.add_argument('--learning-rate', type=float, default=1e-3,
                                help='Learning rate')
    baseline_parser.add_argument('--model-path', type=str, default='models/baseline.pth',
                                help='Path to save trained model')

    # Train SBI model
    sbi_parser = subparsers.add_parser('train-sbi', help='Train SBI model')
    sbi_parser.add_argument('--data-path', type=str, default='data/synthetic_dataset.npz',
                           help='Path to training data')
    sbi_parser.add_argument('--num-simulations', type=int, default=50000,
                           help='Number of simulations for SBI training')
    sbi_parser.add_argument('--model-path', type=str, default='models/sbi_flows.pkl',
                           help='Path to save trained SBI model')

    # Run inference
    infer_parser = subparsers.add_parser('infer', help='Run inference on observational data')
    infer_parser.add_argument('--method', choices=['baseline', 'sbi'], default='sbi',
                             help='Inference method to use')
    infer_parser.add_argument('--model-path', type=str, required=True,
                             help='Path to trained model')
    infer_parser.add_argument('--obs-data-path', type=str, required=True,
                             help='Path to observational data')
    infer_parser.add_argument('--output-dir', type=str, default='results/',
                             help='Output directory for results')

    args = parser.parse_args()

    if args.command == 'generate':
        generate_synthetic_data(args)
    elif args.command == 'train-baseline':
        train_baseline(args)
    elif args.command == 'train-sbi':
        train_sbi(args)
    elif args.command == 'infer':
        run_inference(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
