"""
Plotting utilities for visualizing EoS inference results.
Creates publication-quality figures for posterior analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

class EoSPlotter:
    """Main plotting class for EoS inference results."""

    def __init__(self, figsize_scale=1.0):
        """
        Initialize plotter.

        Parameters:
        -----------
        figsize_scale : float
            Scale factor for figure sizes
        """
        self.figsize_scale = figsize_scale
        self.colors = sns.color_palette("husl", 8)

        # Set matplotlib parameters for publication quality
        plt.rcParams.update({
            'font.size': 12 * figsize_scale,
            'axes.labelsize': 14 * figsize_scale,
            'axes.titlesize': 16 * figsize_scale,
            'xtick.labelsize': 11 * figsize_scale,
            'ytick.labelsize': 11 * figsize_scale,
            'legend.fontsize': 12 * figsize_scale,
            'figure.titlesize': 18 * figsize_scale,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
        })

def create_corner_plot(samples, param_names=None, title="EoS Parameter Posterior", 
                      save_path=None, show_quantiles=True, **kwargs):
    """
    Create corner plot of posterior samples.

    Parameters:
    -----------
    samples : array_like
        Posterior samples (n_samples, n_params)
    param_names : list, optional
        Parameter names for labels
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    show_quantiles : bool
        Show quantile lines on histograms

    Returns:
    --------
    matplotlib.figure.Figure : The corner plot figure
    """
    try:
        import corner
        corner_available = True
    except ImportError:
        corner_available = False

    samples = np.array(samples)
    n_samples, n_dims = samples.shape

    if param_names is None:
        param_names = [f'$\theta_{{{i+1}}}$' for i in range(n_dims)]

    if corner_available:
        # Use corner package if available
        quantiles = [0.16, 0.5, 0.84] if show_quantiles else None

        fig = corner.corner(
            samples,
            labels=param_names,
            quantiles=quantiles,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            **kwargs
        )

        if title:
            fig.suptitle(title, fontsize=16, y=0.98)

    else:
        # Manual corner plot implementation
        fig, axes = plt.subplots(n_dims, n_dims, figsize=(12, 12))

        for i in range(n_dims):
            for j in range(n_dims):
                ax = axes[i, j]

                if i < j:
                    # Upper triangle: hide
                    ax.set_visible(False)

                elif i == j:
                    # Diagonal: histograms
                    ax.hist(samples[:, i], bins=30, density=True, alpha=0.7, 
                           color=sns.color_palette()[0])

                    if show_quantiles:
                        quantiles = np.percentile(samples[:, i], [16, 50, 84])
                        for q in quantiles:
                            ax.axvline(q, color='red', linestyle='--', alpha=0.7)

                    ax.set_xlabel(param_names[i])
                    if i == 0:
                        ax.set_ylabel('Density')

                else:
                    # Lower triangle: 2D histograms
                    ax.hist2d(samples[:, j], samples[:, i], bins=20, density=True, cmap='Blues')

                    ax.set_xlabel(param_names[j])
                    if j == 0:
                        ax.set_ylabel(param_names[i])

        if title:
            fig.suptitle(title, fontsize=16)

        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Corner plot saved to {save_path}")

    return fig

def create_mass_radius_plot(samples, eos_type='piecewise_polytrope', 
                           title="Mass-Radius Relations", save_path=None,
                           n_curves=50, show_observations=True):
    """
    Create mass-radius relation plot from posterior samples.

    Parameters:
    -----------
    samples : array_like
        Posterior samples
    eos_type : str
        Type of EoS parameterization
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    n_curves : int
        Number of M-R curves to plot
    show_observations : bool
        Show observational constraints

    Returns:
    --------
    matplotlib.figure.Figure : The M-R plot figure
    """
    from ..eos.eos_param import PiecewisePolytrope
    from ..physics.tov import TOVSolver

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample random subset for plotting
    n_samples = len(samples)
    plot_indices = np.random.choice(n_samples, min(n_curves, n_samples), replace=False)

    masses_all = []
    radii_all = []

    for idx in plot_indices:
        params = samples[idx]

        try:
            if eos_type == 'piecewise_polytrope':
                eos = PiecewisePolytrope(*params)
            else:
                continue

            solver = TOVSolver(eos)
            masses, radii = solver.mass_radius_relation()

            if len(masses) > 0:
                ax.plot(radii, masses, 'b-', alpha=0.1, linewidth=0.8)
                masses_all.extend(masses)
                radii_all.extend(radii)

        except Exception:
            continue

    # Plot confidence regions
    if len(masses_all) > 0:
        masses_all = np.array(masses_all)
        radii_all = np.array(radii_all)

        # Create grid for confidence contours
        r_grid = np.linspace(8, 16, 50)
        m_percentiles = []

        for r in r_grid:
            # Find masses within radius range
            mask = (radii_all >= r - 0.2) & (radii_all <= r + 0.2)
            if np.sum(mask) > 10:
                m_range = masses_all[mask]
                percentiles = np.percentile(m_range, [5, 16, 50, 84, 95])
                m_percentiles.append(percentiles)
            else:
                m_percentiles.append([np.nan] * 5)

        m_percentiles = np.array(m_percentiles)

        # Plot confidence bands
        valid = ~np.isnan(m_percentiles[:, 2])
        if np.any(valid):
            ax.fill_between(r_grid[valid], m_percentiles[valid, 0], m_percentiles[valid, 4],
                           alpha=0.2, color='blue', label='95% confidence')
            ax.fill_between(r_grid[valid], m_percentiles[valid, 1], m_percentiles[valid, 3],
                           alpha=0.4, color='blue', label='68% confidence')
            ax.plot(r_grid[valid], m_percentiles[valid, 2], 'b-', linewidth=2, label='Median')

    # Add observational constraints
    if show_observations:
        # NICER constraints (approximate)
        nicer_data = [
            {"name": "PSR J0030+0451", "mass": 1.44, "mass_err": 0.15, 
             "radius": 12.7, "radius_err": 1.1, "color": "red"},
            {"name": "PSR J0740+6620", "mass": 2.08, "mass_err": 0.07,
             "radius": 12.4, "radius_err": 1.3, "color": "green"}
        ]

        for pulsar in nicer_data:
            ax.errorbar(pulsar["radius"], pulsar["mass"], 
                       xerr=pulsar["radius_err"], yerr=pulsar["mass_err"],
                       fmt='o', color=pulsar["color"], markersize=8, 
                       capsize=4, capthick=2, label=pulsar["name"])

        # Heavy pulsar constraints
        heavy_pulsars = [
            {"name": "PSR J1614-2230", "mass": 1.97, "mass_err": 0.04},
            {"name": "PSR J0348+0432", "mass": 2.01, "mass_err": 0.04}
        ]

        for pulsar in heavy_pulsars:
            ax.axhline(y=pulsar["mass"], color='gray', linestyle='--', alpha=0.7)
            ax.fill_between([8, 18], 
                           pulsar["mass"] - pulsar["mass_err"],
                           pulsar["mass"] + pulsar["mass_err"], 
                           alpha=0.1, color='gray')

    ax.set_xlabel('Radius (km)')
    ax.set_ylabel('Mass (M☉)')
    ax.set_title(title)
    ax.set_xlim(8, 18)
    ax.set_ylim(0.5, 2.8)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mass-radius plot saved to {save_path}")

    return fig

def create_eos_plot(samples, eos_type='piecewise_polytrope', 
                   title="Equation of State", save_path=None, n_curves=50):
    """
    Plot pressure vs density for EoS posterior samples.

    Parameters:
    -----------
    samples : array_like
        Posterior samples
    eos_type : str
        EoS parameterization type
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    n_curves : int
        Number of EoS curves to plot

    Returns:
    --------
    matplotlib.figure.Figure : The EoS plot figure
    """
    from ..eos.eos_param import PiecewisePolytrope
    from ..eos.units import RHO_NUC

    fig, ax = plt.subplots(figsize=(10, 8))

    # Density range
    rho_range = np.logspace(np.log10(0.1 * RHO_NUC), np.log10(10 * RHO_NUC), 100)
    rho_norm = rho_range / RHO_NUC  # Normalize by nuclear density

    # Sample random subset for plotting
    n_samples = len(samples)
    plot_indices = np.random.choice(n_samples, min(n_curves, n_samples), replace=False)

    pressure_curves = []

    for idx in plot_indices:
        params = samples[idx]

        try:
            if eos_type == 'piecewise_polytrope':
                eos = PiecewisePolytrope(*params)
            else:
                continue

            pressure = eos.pressure(rho_range)

            # Only plot if physically reasonable
            if np.all(pressure >= 0) and not np.any(np.isnan(pressure)):
                ax.loglog(rho_norm, pressure, 'b-', alpha=0.1, linewidth=0.8)
                pressure_curves.append(pressure)

        except Exception:
            continue

    # Plot confidence bands
    if len(pressure_curves) > 0:
        pressure_curves = np.array(pressure_curves)

        percentiles = np.percentile(pressure_curves, [5, 16, 50, 84, 95], axis=0)

        ax.fill_between(rho_norm, percentiles[0], percentiles[4],
                       alpha=0.2, color='blue', label='95% confidence')
        ax.fill_between(rho_norm, percentiles[1], percentiles[3],
                       alpha=0.4, color='blue', label='68% confidence')
        ax.loglog(rho_norm, percentiles[2], 'b-', linewidth=2, label='Median')

    # Add causality limit (P = ρ c²)
    ax.loglog(rho_norm, rho_range, 'k--', alpha=0.7, label='Causality limit')

    # Add nuclear saturation density line
    ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.7, 
              label='Nuclear saturation density')

    ax.set_xlabel('Density (ρ/ρ₀)')
    ax.set_ylabel('Pressure (GeV/fm³)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"EoS plot saved to {save_path}")

    return fig

def create_tidal_deformability_plot(samples, eos_type='piecewise_polytrope',
                                   title="Tidal Deformability", save_path=None,
                                   n_curves=50, show_gw_constraint=True):
    """
    Plot tidal deformability vs mass from posterior samples.

    Parameters:
    -----------
    samples : array_like
        Posterior samples
    eos_type : str
        EoS parameterization type
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    n_curves : int
        Number of curves to plot
    show_gw_constraint : bool
        Show GW170817 constraint

    Returns:
    --------
    matplotlib.figure.Figure : The tidal plot figure
    """
    from ..eos.eos_param import PiecewisePolytrope
    from ..physics.tov import TOVSolver

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sample random subset
    n_samples = len(samples)
    plot_indices = np.random.choice(n_samples, min(n_curves, n_samples), replace=False)

    lambda_curves = []
    mass_range = np.linspace(1.0, 2.5, 50)

    for idx in plot_indices:
        params = samples[idx]

        try:
            if eos_type == 'piecewise_polytrope':
                eos = PiecewisePolytrope(*params)
            else:
                continue

            solver = TOVSolver(eos)

            # Calculate Lambda for different masses
            lambda_values = []
            valid_masses = []

            for mass in mass_range:
                try:
                    lambda_val = solver.calculate_tidal_deformability(mass)
                    if lambda_val is not None and lambda_val > 0:
                        lambda_values.append(lambda_val)
                        valid_masses.append(mass)
                except:
                    continue

            if len(lambda_values) > 5:  # Minimum points for a curve
                ax.loglog(valid_masses, lambda_values, 'b-', alpha=0.1, linewidth=0.8)

                # Interpolate to common mass grid
                if len(valid_masses) > 0:
                    lambda_interp = np.interp(mass_range, valid_masses, lambda_values, 
                                            left=np.nan, right=np.nan)
                    lambda_curves.append(lambda_interp)

        except Exception:
            continue

    # Plot confidence bands
    if len(lambda_curves) > 0:
        lambda_curves = np.array(lambda_curves)

        # Calculate percentiles (ignoring NaN values)
        percentiles = np.nanpercentile(lambda_curves, [5, 16, 50, 84, 95], axis=0)

        valid_idx = ~np.isnan(percentiles[2])

        ax.fill_between(mass_range[valid_idx], percentiles[0, valid_idx], 
                       percentiles[4, valid_idx],
                       alpha=0.2, color='blue', label='95% confidence')
        ax.fill_between(mass_range[valid_idx], percentiles[1, valid_idx], 
                       percentiles[3, valid_idx],
                       alpha=0.4, color='blue', label='68% confidence')
        ax.loglog(mass_range[valid_idx], percentiles[2, valid_idx], 
                 'b-', linewidth=2, label='Median')

    # Add GW170817 constraint
    if show_gw_constraint:
        # Approximate constraint from GW170817
        gw_mass = 1.4
        gw_lambda = 190
        gw_lambda_err = 120

        ax.errorbar(gw_mass, gw_lambda, yerr=[[gw_lambda_err], [gw_lambda_err]], 
                   fmt='ro', markersize=8, capsize=4, capthick=2, 
                   label='GW170817 (approximate)')

        # Show constraint region
        ax.fill_between([1.35, 1.45], [70, 70], [580, 580], 
                       alpha=0.2, color='red', label='GW170817 constraint')

    ax.set_xlabel('Mass (M☉)')
    ax.set_ylabel('Tidal Deformability (Λ)')
    ax.set_title(title)
    ax.set_xlim(1.0, 2.5)
    ax.set_ylim(1, 5000)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Tidal deformability plot saved to {save_path}")

    return fig

def create_parameter_evolution_plot(samples, param_names=None, 
                                   title="Parameter Evolution", save_path=None):
    """
    Plot evolution of parameters during sampling (trace plots).

    Parameters:
    -----------
    samples : array_like
        Samples with shape (n_steps, n_walkers, n_params) or (n_steps, n_params)
    param_names : list, optional
        Parameter names
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure : The trace plot figure
    """
    samples = np.array(samples)

    if samples.ndim == 3:
        # Multiple walkers: (n_steps, n_walkers, n_params)
        n_steps, n_walkers, n_params = samples.shape
    elif samples.ndim == 2:
        # Single chain: (n_steps, n_params)
        n_steps, n_params = samples.shape
        n_walkers = 1
        samples = samples[:, None, :]  # Add walker dimension
    else:
        raise ValueError("Samples must be 2D or 3D array")

    if param_names is None:
        param_names = [f'$\theta_{{{i+1}}}$' for i in range(n_params)]

    fig, axes = plt.subplots(n_params, 1, figsize=(12, 2 * n_params), sharex=True)

    if n_params == 1:
        axes = [axes]

    steps = np.arange(n_steps)

    for i, (ax, param_name) in enumerate(zip(axes, param_names)):
        # Plot all walkers
        for walker in range(n_walkers):
            ax.plot(steps, samples[:, walker, i], alpha=0.5, linewidth=0.8)

        ax.set_ylabel(param_name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Step')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter evolution plot saved to {save_path}")

    return fig

def create_comparison_plot(samples_list, labels, param_names=None,
                          title="Model Comparison", save_path=None):
    """
    Compare posterior distributions from different models.

    Parameters:
    -----------
    samples_list : list of array_like
        List of posterior sample arrays
    labels : list of str
        Labels for each sample set
    param_names : list, optional
        Parameter names
    title : str
        Plot title
    save_path : str, optional
        Path to save figure

    Returns:
    --------
    matplotlib.figure.Figure : The comparison plot figure
    """
    n_models = len(samples_list)
    n_params = samples_list[0].shape[1]

    if param_names is None:
        param_names = [f'$\theta_{{{i+1}}}$' for i in range(n_params)]

    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8))

    if n_params == 1:
        axes = axes.reshape(2, 1)

    colors = sns.color_palette("husl", n_models)

    for param_idx in range(n_params):
        # Histograms
        ax_hist = axes[0, param_idx]

        for model_idx, (samples, label, color) in enumerate(zip(samples_list, labels, colors)):
            ax_hist.hist(samples[:, param_idx], bins=30, alpha=0.6, 
                        label=label, color=color, density=True)

        ax_hist.set_xlabel(param_names[param_idx])
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        # Box plots
        ax_box = axes[1, param_idx]

        data_for_boxplot = [samples[:, param_idx] for samples in samples_list]
        box_plot = ax_box.boxplot(data_for_boxplot, labels=labels, patch_artist=True)

        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax_box.set_xlabel('Model')
        ax_box.set_ylabel(param_names[param_idx])
        ax_box.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    return fig

def save_all_plots(samples, output_dir='results/plots', eos_type='piecewise_polytrope',
                  param_names=None, prefix='eos_inference'):
    """
    Generate and save all standard plots for EoS inference results.

    Parameters:
    -----------
    samples : array_like
        Posterior samples
    output_dir : str
        Output directory for plots
    eos_type : str
        EoS parameterization type
    param_names : list, optional
        Parameter names
    prefix : str
        Filename prefix

    Returns:
    --------
    dict : Dictionary of saved plot paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_plots = {}

    # Corner plot
    corner_path = output_dir / f'{prefix}_corner.png'
    fig_corner = create_corner_plot(samples, param_names=param_names, 
                                   save_path=corner_path)
    plt.close(fig_corner)
    saved_plots['corner'] = corner_path

    # Mass-radius plot
    mr_path = output_dir / f'{prefix}_mass_radius.png'
    fig_mr = create_mass_radius_plot(samples, eos_type=eos_type, 
                                    save_path=mr_path)
    plt.close(fig_mr)
    saved_plots['mass_radius'] = mr_path

    # EoS plot
    eos_path = output_dir / f'{prefix}_eos.png'
    fig_eos = create_eos_plot(samples, eos_type=eos_type, 
                             save_path=eos_path)
    plt.close(fig_eos)
    saved_plots['eos'] = eos_path

    # Tidal deformability plot
    tidal_path = output_dir / f'{prefix}_tidal.png'
    fig_tidal = create_tidal_deformability_plot(samples, eos_type=eos_type,
                                               save_path=tidal_path)
    plt.close(fig_tidal)
    saved_plots['tidal'] = tidal_path

    print(f"All plots saved to {output_dir}")
    return saved_plots
