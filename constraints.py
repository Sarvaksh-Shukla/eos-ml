"""
Physical constraints on neutron star equation of state.
Implements causality, thermodynamic stability, and observational constraints.
"""

import numpy as np
from scipy.optimize import brentq
from .units import RHO_NUC, Units

class EoSConstraints:
    """Class to check physical constraints on equation of state."""

    @staticmethod
    def check_causality(eos, density_range=None):
        """Check if equation of state satisfies causality (c_s^2 <= 1)."""
        if density_range is None:
            density_range = np.logspace(
                np.log10(0.5 * RHO_NUC), 
                np.log10(10 * RHO_NUC), 
                100
            )

        cs2_values = eos.speed_of_sound_squared(density_range)

        # Check if any values exceed speed of light
        causality_violated = np.any(cs2_values > 1.0)
        max_cs2 = np.max(cs2_values) if len(cs2_values) > 0 else 0

        return {
            'satisfied': not causality_violated,
            'max_cs2': max_cs2,
            'violation_density': density_range[np.argmax(cs2_values)] if causality_violated else None
        }

    @staticmethod
    def check_thermodynamic_stability(eos, density_range=None):
        """Check thermodynamic stability (dp/drho >= 0)."""
        if density_range is None:
            density_range = np.logspace(
                np.log10(0.5 * RHO_NUC), 
                np.log10(10 * RHO_NUC), 
                100
            )

        # Calculate dp/drho numerically
        h = 1e-8
        dp_drho = np.gradient(eos.pressure(density_range), density_range)

        # Check if any derivatives are negative
        stability_violated = np.any(dp_drho < 0)
        min_dp_drho = np.min(dp_drho)

        return {
            'satisfied': not stability_violated,
            'min_dp_drho': min_dp_drho,
            'violation_density': density_range[np.argmin(dp_drho)] if stability_violated else None
        }

    @staticmethod
    def check_low_density_matching(eos, reference_pressure=None):
        """Check if EoS matches known low-density nuclear physics."""
        rho_match = 0.5 * RHO_NUC  # Matching density

        if reference_pressure is None:
            # Use approximate nuclear matter calculation
            reference_pressure = 0.1  # GeV/fm^3 (rough estimate)

        eos_pressure = eos.pressure(rho_match)
        relative_deviation = abs(eos_pressure - reference_pressure) / reference_pressure

        return {
            'satisfied': relative_deviation < 0.5,  # Allow 50% deviation
            'relative_deviation': relative_deviation,
            'eos_pressure': eos_pressure,
            'reference_pressure': reference_pressure
        }

    @staticmethod
    def check_high_density_behavior(eos):
        """Check reasonable high-density behavior."""
        rho_high = 10 * RHO_NUC

        # Pressure should be substantial at high density
        p_high = eos.pressure(rho_high)

        # Check if pressure is reasonable (not too small or too large)
        reasonable_pressure = 1.0 < p_high < 100.0  # GeV/fm^3

        return {
            'satisfied': reasonable_pressure,
            'high_density_pressure': p_high,
            'density': rho_high
        }

    @classmethod
    def check_all_constraints(cls, eos, verbose=False):
        """Check all physical constraints on the EoS."""
        results = {}

        # Causality
        results['causality'] = cls.check_causality(eos)

        # Thermodynamic stability
        results['stability'] = cls.check_thermodynamic_stability(eos)

        # Low-density matching
        results['low_density'] = cls.check_low_density_matching(eos)

        # High-density behavior
        results['high_density'] = cls.check_high_density_behavior(eos)

        # Overall satisfaction
        all_satisfied = all(result['satisfied'] for result in results.values())
        results['all_satisfied'] = all_satisfied

        if verbose:
            print(f"Constraint check for {eos.name}:")
            print(f"  Causality: {'✓' if results['causality']['satisfied'] else '✗'}")
            if not results['causality']['satisfied']:
                print(f"    Max c_s^2 = {results['causality']['max_cs2']:.3f}")

            print(f"  Stability: {'✓' if results['stability']['satisfied'] else '✗'}")
            if not results['stability']['satisfied']:
                print(f"    Min dp/drho = {results['stability']['min_dp_drho']:.3e}")

            print(f"  Low-density: {'✓' if results['low_density']['satisfied'] else '✗'}")
            if not results['low_density']['satisfied']:
                print(f"    Deviation = {results['low_density']['relative_deviation']:.1%}")

            print(f"  High-density: {'✓' if results['high_density']['satisfied'] else '✗'}")
            print(f"  Overall: {'✓' if all_satisfied else '✗'}")

        return results

class ObservationalConstraints:
    """Observational constraints from neutron star observations."""

    # Pulsar mass measurements (solar masses)
    PULSAR_MASSES = {
        'J0348+0432': 2.01,
        'J0740+6620': 2.14,
        'J1614-2230': 1.97,
        'J1903+0327': 1.67
    }

    # NICER radius constraints (km)
    NICER_RADII = {
        'J0030+0451': (12.71, 1.14),  # (central, uncertainty)
        'J0740+6620': (12.39, 1.30)
    }

    # GW170817 tidal deformability constraint
    GW170817_LAMBDA_1400 = (190, 120)  # (central, uncertainty) for 1.4 Msun

    @classmethod
    def check_mass_constraint(cls, mass_radius_relation, min_max_mass=2.0):
        """Check if EoS can support observed heavy pulsars."""
        masses, radii = mass_radius_relation
        max_mass = np.max(masses)

        return {
            'satisfied': max_mass >= min_max_mass,
            'max_mass': max_mass,
            'required_mass': min_max_mass
        }

    @classmethod
    def check_radius_constraint(cls, mass_radius_relation, 
                              target_mass=1.4, radius_range=(11.0, 14.0)):
        """Check radius constraint for canonical 1.4 solar mass NS."""
        masses, radii = mass_radius_relation

        # Find radius at target mass
        if target_mass in masses:
            idx = np.where(masses == target_mass)[0][0]
            radius_14 = radii[idx]
        else:
            # Interpolate
            radius_14 = np.interp(target_mass, masses, radii)

        satisfied = radius_range[0] <= radius_14 <= radius_range[1]

        return {
            'satisfied': satisfied,
            'radius_14': radius_14,
            'radius_range': radius_range
        }

    @classmethod
    def check_tidal_constraint(cls, tidal_deformability, 
                             target_mass=1.4, lambda_range=(70, 580)):
        """Check tidal deformability constraint from GW170817."""
        masses, lambdas = tidal_deformability

        # Find Lambda at target mass
        if target_mass in masses:
            idx = np.where(masses == target_mass)[0][0]
            lambda_14 = lambdas[idx]
        else:
            # Interpolate in log space for Lambda
            log_lambda = np.interp(target_mass, masses, np.log(lambdas))
            lambda_14 = np.exp(log_lambda)

        satisfied = lambda_range[0] <= lambda_14 <= lambda_range[1]

        return {
            'satisfied': satisfied,
            'lambda_14': lambda_14,
            'lambda_range': lambda_range
        }

    @classmethod
    def check_all_observational_constraints(cls, mass_radius_relation, 
                                          tidal_deformability=None, verbose=False):
        """Check all observational constraints."""
        results = {}

        # Maximum mass constraint
        results['max_mass'] = cls.check_mass_constraint(mass_radius_relation)

        # Radius constraint
        results['radius'] = cls.check_radius_constraint(mass_radius_relation)

        # Tidal constraint (if provided)
        if tidal_deformability is not None:
            results['tidal'] = cls.check_tidal_constraint(tidal_deformability)

        # Overall satisfaction
        all_satisfied = all(result['satisfied'] for result in results.values())
        results['all_satisfied'] = all_satisfied

        if verbose:
            print("Observational constraint check:")
            print(f"  Max mass: {'✓' if results['max_mass']['satisfied'] else '✗'}")
            print(f"    M_max = {results['max_mass']['max_mass']:.2f} M_sun")

            print(f"  Radius: {'✓' if results['radius']['satisfied'] else '✗'}")
            print(f"    R_1.4 = {results['radius']['radius_14']:.1f} km")

            if 'tidal' in results:
                print(f"  Tidal: {'✓' if results['tidal']['satisfied'] else '✗'}")
                print(f"    Λ_1.4 = {results['tidal']['lambda_14']:.0f}")

            print(f"  Overall: {'✓' if all_satisfied else '✗'}")

        return results
