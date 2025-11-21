"""
Physical units and constants for neutron star calculations.
Uses Planck units (c = G = hbar = 1) and converts to/from SI units.
"""

import numpy as np

# Fundamental constants in SI units
C_LIGHT = 2.99792458e8  # m/s
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2
HBAR = 1.054571817e-34  # J s
M_SUN = 1.98847e30      # kg
R_SUN = 6.96e8          # m

# Conversion factors
# Energy density: GeV/fm^3 to kg/m^3
ENERGY_DENSITY_CONVERSION = 1.783e17  # kg/m^3 per GeV/fm^3

# Pressure: GeV/fm^3 to Pa
PRESSURE_CONVERSION = 1.602176634e35  # Pa per GeV/fm^3

# Length: km to m
KM_TO_M = 1000.0

class Units:
    """Unit conversion utilities for neutron star physics."""

    @staticmethod
    def energy_density_to_si(rho_gev_fm3):
        """Convert energy density from GeV/fm^3 to kg/m^3."""
        return rho_gev_fm3 * ENERGY_DENSITY_CONVERSION

    @staticmethod
    def pressure_to_si(p_gev_fm3):
        """Convert pressure from GeV/fm^3 to Pa."""
        return p_gev_fm3 * PRESSURE_CONVERSION

    @staticmethod
    def mass_to_solar(mass_kg):
        """Convert mass from kg to solar masses."""
        return mass_kg / M_SUN

    @staticmethod
    def radius_to_km(radius_m):
        """Convert radius from m to km."""
        return radius_m / KM_TO_M

    @staticmethod
    def get_nuclear_saturation_density():
        """Nuclear saturation density in GeV/fm^3."""
        return 0.16  # GeV/fm^3

    @staticmethod
    def get_nuclear_saturation_density_si():
        """Nuclear saturation density in kg/m^3."""
        return Units.energy_density_to_si(0.16)

# Commonly used values
RHO_NUC = Units.get_nuclear_saturation_density()  # GeV/fm^3
RHO_NUC_SI = Units.get_nuclear_saturation_density_si()  # kg/m^3

# Typical neutron star parameter ranges
NS_MASS_RANGE = (1.0, 2.8)  # Solar masses
NS_RADIUS_RANGE = (8.0, 16.0)  # km
NS_CENTRAL_DENSITY_RANGE = (2.0, 15.0)  # times nuclear density
