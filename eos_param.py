"""
Equation of state parameterizations for neutron star matter.
Implements various EoS models including piecewise polytropes and spectral decomposition.
"""

import numpy as np
from scipy.interpolate import interp1d
from .units import RHO_NUC, Units

class EoSParameterization:
    """Base class for equation of state parameterizations."""

    def __init__(self, name="Generic EoS"):
        self.name = name
        self.parameters = {}

    def pressure(self, density):
        """Return pressure as function of density."""
        raise NotImplementedError

    def energy_density(self, density):
        """Return energy density as function of density."""
        raise NotImplementedError

    def speed_of_sound_squared(self, density):
        """Return squared speed of sound c_s^2 = dp/d(energy_density)."""
        # Numerical derivative
        h = 1e-8
        rho_plus = density + h
        rho_minus = density - h

        p_plus = self.pressure(rho_plus)
        p_minus = self.pressure(rho_minus)
        e_plus = self.energy_density(rho_plus)
        e_minus = self.energy_density(rho_minus)

        dp_drho = (p_plus - p_minus) / (2 * h)
        de_drho = (e_plus - e_minus) / (2 * h)

        return dp_drho / de_drho

class PiecewisePolytrope(EoSParameterization):
    """Piecewise polytropic equation of state."""

    def __init__(self, log_p1, gamma1, gamma2, gamma3, 
                 rho1=RHO_NUC, rho2=2*RHO_NUC, rho3=6*RHO_NUC):
        super().__init__("Piecewise Polytrope")

        self.log_p1 = log_p1  # log10(P1/dyn cm^-2)
        self.gamma1 = gamma1  # First polytropic index
        self.gamma2 = gamma2  # Second polytropic index  
        self.gamma3 = gamma3  # Third polytropic index

        self.rho1 = rho1  # First transition density
        self.rho2 = rho2  # Second transition density
        self.rho3 = rho3  # Third transition density

        self.parameters = {
            'log_p1': log_p1,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'gamma3': gamma3
        }

        # Calculate pressure normalizations for continuity
        self.p1 = 10**log_p1 * 1e-1  # Convert dyn/cm^2 to GeV/fm^3
        self.p2 = self.p1 * (rho2/rho1)**gamma1
        self.p3 = self.p2 * (rho3/rho2)**gamma2

    def pressure(self, density):
        """Calculate pressure from density."""
        density = np.atleast_1d(density)
        pressure = np.zeros_like(density)

        # Region 1: rho < rho1
        mask1 = density < self.rho1
        pressure[mask1] = self.p1 * (density[mask1] / self.rho1)**self.gamma1

        # Region 2: rho1 <= rho < rho2
        mask2 = (density >= self.rho1) & (density < self.rho2)
        pressure[mask2] = self.p2 * (density[mask2] / self.rho2)**self.gamma2

        # Region 3: rho2 <= rho < rho3
        mask3 = (density >= self.rho2) & (density < self.rho3)
        pressure[mask3] = self.p3 * (density[mask3] / self.rho3)**self.gamma3

        # Region 4: rho >= rho3 (extrapolate with gamma3)
        mask4 = density >= self.rho3
        pressure[mask4] = self.p3 * (density[mask4] / self.rho3)**self.gamma3

        return pressure.squeeze() if pressure.shape == (1,) else pressure

    def energy_density(self, density):
        """Calculate energy density from density (assumes E = rho c^2)."""
        return density  # In natural units where c = 1

class SpectralEoS(EoSParameterization):
    """Spectral decomposition equation of state."""

    def __init__(self, coefficients, n_poly=4):
        super().__init__("Spectral EoS")
        self.coefficients = np.array(coefficients)
        self.n_poly = n_poly
        self.parameters = {f'c_{i}': c for i, c in enumerate(coefficients)}

    def _log_enthalpy(self, density):
        """Calculate log of specific enthalpy."""
        x = np.log(density / RHO_NUC)

        # Spectral expansion
        h = 0
        for i, c in enumerate(self.coefficients):
            h += c * x**i

        return h

    def pressure(self, density):
        """Calculate pressure from spectral representation."""
        log_h = self._log_enthalpy(density)
        # P = rho * exp(h) - rho (from thermodynamic identity)
        return density * (np.exp(log_h) - 1)

    def energy_density(self, density):
        """Calculate energy density."""
        log_h = self._log_enthalpy(density)
        return density * np.exp(log_h)

class TabularEoS(EoSParameterization):
    """Tabular equation of state with interpolation."""

    def __init__(self, density_table, pressure_table, energy_table=None):
        super().__init__("Tabular EoS")

        self.density_table = np.array(density_table)
        self.pressure_table = np.array(pressure_table)

        if energy_table is None:
            self.energy_table = density_table  # Default to E = rho c^2
        else:
            self.energy_table = np.array(energy_table)

        # Create interpolation functions
        self._pressure_interp = interp1d(
            self.density_table, self.pressure_table,
            kind='cubic', bounds_error=False, fill_value='extrapolate'
        )

        self._energy_interp = interp1d(
            self.density_table, self.energy_table,
            kind='cubic', bounds_error=False, fill_value='extrapolate'
        )

    def pressure(self, density):
        """Interpolate pressure from table."""
        return self._pressure_interp(density)

    def energy_density(self, density):
        """Interpolate energy density from table."""
        return self._energy_interp(density)

def create_random_piecewise_polytr(rng=None):
    """Create a random piecewise polytrope within physical bounds."""
    if rng is None:
        rng = np.random.default_rng()

    # Sample parameters from reasonable ranges
    log_p1 = rng.uniform(33.5, 35.5)  # log10(P1/dyn cm^-2)
    gamma1 = rng.uniform(1.5, 4.5)
    gamma2 = rng.uniform(1.5, 4.5) 
    gamma3 = rng.uniform(1.5, 4.5)

    return PiecewisePolytrope(log_p1, gamma1, gamma2, gamma3)

def create_random_spectral_eos(n_coeffs=4, rng=None):
    """Create a random spectral EoS."""
    if rng is None:
        rng = np.random.default_rng()

    # Sample coefficients (first coefficient should be around ln(2) ≈ 0.693)
    coefficients = [rng.normal(0.693, 0.1)]  # c_0
    for i in range(1, n_coeffs):
        coefficients.append(rng.normal(0, 0.5))

    return SpectralEoS(coefficients)
