"""
Tolman-Oppenheimer-Volkoff (TOV) equation solver for neutron star structure.
Computes mass-radius relations and other stellar properties from equation of state.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import warnings

class TOVSolver:
    """Solver for the TOV equations in general relativity."""

    def __init__(self, eos, max_radius=20.0, rtol=1e-8):
        """
        Initialize TOV solver.

        Parameters:
        -----------
        eos : EoSParameterization
            Equation of state object
        max_radius : float
            Maximum integration radius in km
        rtol : float
            Relative tolerance for ODE integration
        """
        self.eos = eos
        self.max_radius = max_radius  # km
        self.rtol = rtol

        # Conversion factors (geometric units)
        self.G = 6.67430e-11  # m^3 kg^-1 s^-2
        self.c = 2.99792458e8  # m/s
        self.M_sun = 1.98847e30  # kg

        # Convert to geometric units: G = c = 1
        # Length scale: GM_sun/c^2 = 1.477 km
        self.length_scale = self.G * self.M_sun / (self.c**2) / 1000  # km

        # Pressure/density scale: c^2 = 9e16 m^2/s^2
        self.pressure_scale = 1.602176634e35  # Pa per GeV/fm^3 -> geometric

    def tov_equations(self, r, y, central_density):
        """
        TOV differential equations.

        y = [m, p] where:
        m = mass enclosed within radius r
        p = pressure at radius r
        """
        m, p = y

        if p <= 0:
            return [0, 0]  # Stop integration at surface

        try:
            # Get energy density from pressure via EoS
            rho = self.eos.energy_density(p)

            # TOV equations in geometric units
            dm_dr = 4 * np.pi * r**2 * rho

            if r > 0:
                dp_dr = -(rho + p) * (m + 4 * np.pi * r**3 * p) / (r * (r - 2*m))
            else:
                dp_dr = 0  # Central condition

        except (ValueError, ZeroDivisionError, RuntimeWarning):
            return [0, 0]

        return [dm_dr, dp_dr]

    def solve_structure(self, central_density, r_max=None):
        """
        Solve stellar structure for given central density.

        Parameters:
        -----------
        central_density : float
            Central density in GeV/fm^3
        r_max : float
            Maximum radius for integration in km

        Returns:
        --------
        dict : Solution containing radius, mass, pressure, density profiles
        """
        if r_max is None:
            r_max = self.max_radius

        # Initial conditions
        central_pressure = self.eos.pressure(central_density)

        if central_pressure <= 0:
            return None

        # Small initial radius to avoid singularity
        r0 = 1e-6  # km
        m0 = (4/3) * np.pi * r0**3 * central_density
        y0 = [m0, central_pressure]

        # Integration points
        r_span = (r0, r_max)

        # Event function to stop at surface (p = 0)
        def surface_event(r, y):
            return y[1]  # pressure
        surface_event.terminal = True
        surface_event.direction = -1

        try:
            # Solve TOV equations
            sol = solve_ivp(
                lambda r, y: self.tov_equations(r, y, central_density),
                r_span, y0,
                events=surface_event,
                rtol=self.rtol,
                atol=1e-10,
                method='DOP853',
                max_step=0.01
            )

            if not sol.success or len(sol.y[0]) < 2:
                return None

            # Extract solution
            radius = sol.t
            mass = sol.y[0]
            pressure = sol.y[1]

            # Calculate density profile
            density = np.zeros_like(pressure)
            for i, p in enumerate(pressure):
                if p > 0:
                    try:
                        density[i] = self.eos.energy_density(p)
                    except:
                        density[i] = 0
                else:
                    density[i] = 0

            # Stellar radius and mass
            stellar_radius = radius[-1]  # km
            stellar_mass = mass[-1]     # solar masses

            return {
                'radius_profile': radius,
                'mass_profile': mass,
                'pressure_profile': pressure,
                'density_profile': density,
                'stellar_radius': stellar_radius,
                'stellar_mass': stellar_mass,
                'central_density': central_density,
                'central_pressure': central_pressure
            }

        except Exception as e:
            warnings.warn(f"TOV integration failed: {e}")
            return None

    def mass_radius_relation(self, density_range=None, n_points=100):
        """
        Calculate mass-radius relation by varying central density.

        Parameters:
        -----------
        density_range : tuple
            (min_density, max_density) in units of nuclear density
        n_points : int
            Number of points to calculate

        Returns:
        --------
        tuple : (masses, radii) arrays
        """
        if density_range is None:
            density_range = (0.5, 15.0)  # In units of nuclear density

        from ..eos.units import RHO_NUC

        # Convert to absolute densities
        rho_min = density_range[0] * RHO_NUC
        rho_max = density_range[1] * RHO_NUC

        # Log-spaced central densities
        central_densities = np.logspace(
            np.log10(rho_min), 
            np.log10(rho_max), 
            n_points
        )

        masses = []
        radii = []

        for rho_c in central_densities:
            solution = self.solve_structure(rho_c)

            if solution is not None:
                masses.append(solution['stellar_mass'])
                radii.append(solution['stellar_radius'])

        return np.array(masses), np.array(radii)

    def find_maximum_mass(self, density_range=None):
        """Find the maximum mass configuration."""
        masses, radii = self.mass_radius_relation(density_range)

        if len(masses) == 0:
            return None, None

        max_idx = np.argmax(masses)
        return masses[max_idx], radii[max_idx]

    def calculate_tidal_deformability(self, mass_target=1.4):
        """
        Calculate tidal deformability Lambda for a given mass.

        Parameters:
        -----------
        mass_target : float
            Target mass in solar masses

        Returns:
        --------
        float : Tidal deformability Lambda
        """
        # This is a simplified calculation
        # Full calculation requires solving tidal Love number equations

        masses, radii = self.mass_radius_relation()

        if len(masses) == 0:
            return None

        # Find radius corresponding to target mass
        if mass_target < np.min(masses) or mass_target > np.max(masses):
            return None

        radius = np.interp(mass_target, masses, radii)

        # Approximate formula for tidal deformability
        # Lambda ≈ (2/3) k_2 R^5 / (GM/c^2)^5
        # Using k_2 ≈ 0.1 as rough approximation
        k2 = 0.1  # Love number (approximate)

        # Convert to dimensionless units
        R_dim = radius / self.length_scale  # Dimensionless radius
        M_dim = mass_target  # Already in solar masses

        Lambda = (2/3) * k2 * (R_dim**5) / (M_dim**5)

        # Convert to physical units (roughly)
        Lambda *= 1e36  # Rough conversion factor

        return Lambda

    def get_compactness(self, mass, radius):
        """Calculate gravitational compactness C = GM/(Rc^2)."""
        return (mass * self.length_scale) / radius

class TOVSequence:
    """Generate sequences of neutron star models."""

    def __init__(self, eos_list):
        """
        Initialize with list of equation of state models.

        Parameters:
        -----------
        eos_list : list
            List of EoSParameterization objects
        """
        self.eos_list = eos_list
        self.solvers = [TOVSolver(eos) for eos in eos_list]

    def calculate_mr_relations(self):
        """Calculate M-R relations for all EoS models."""
        results = []

        for i, solver in enumerate(self.solvers):
            masses, radii = solver.mass_radius_relation()

            results.append({
                'eos_name': self.eos_list[i].name,
                'masses': masses,
                'radii': radii,
                'max_mass': np.max(masses) if len(masses) > 0 else 0
            })

        return results

    def filter_by_constraints(self, min_max_mass=2.0, radius_range=(11.0, 14.0)):
        """Filter EoS models by observational constraints."""
        mr_results = self.calculate_mr_relations()

        valid_eos = []
        for i, result in enumerate(mr_results):
            # Check maximum mass constraint
            if result['max_mass'] < min_max_mass:
                continue

            # Check radius constraint for 1.4 solar mass star
            masses, radii = result['masses'], result['radii']
            if len(masses) == 0:
                continue

            try:
                r_14 = np.interp(1.4, masses, radii)
                if not (radius_range[0] <= r_14 <= radius_range[1]):
                    continue
            except:
                continue

            valid_eos.append(self.eos_list[i])

        return valid_eos
