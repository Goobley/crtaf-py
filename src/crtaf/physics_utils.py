from dataclasses import dataclass
from typing import Optional
import astropy.constants as const
import astropy.units as u
import lightweaver as lw
import numpy as np

def n_eff(e_upper, e_lower, Z):
    e_rydberg = const.Ryd.to(u.J, equivalencies=u.spectral())
    h_amu = lw.PeriodicTable['H'].mass
    # NOTE: H ionisation
    ry_h = e_rydberg / (1.0 + const.m_e / (h_amu * const.u))

    e_u = e_upper.to(u.J)
    e_l = e_lower.to(u.J)
    return (Z * np.sqrt(ry_h / (e_u - e_l))).to(u.Unit("J/J"))

def Aji(lambda0, g_ratio, f_value):
    cst = 2.0 * np.pi * (const.e.si / const.eps0) * (const.e.si / const.m_e) / const.c
    return (cst / lambda0**2 * g_ratio * f_value).to(1.0 / u.s)

def Bji(Aji, lambda0):
    return Aji * lambda0**3 / (2.0 * const.h * const.c)

def Bji_lambda(Aji, lambda0):
    return Aji * lambda0**5 / (2.0 * const.h * const.c**2)

def Bij(Bji, g_ratio):
    return g_ratio * Bji

@dataclass
class EinsteinCoeffs:
    Aji: u.Quantity
    Bji: u.Quantity
    Bji_wavelength: u.Quantity
    Bij: u.Quantity
    Bij_wavelength: u.Quantity

    @classmethod
    def compute(cls, lambda0: u.Quantity[u.m], g_ratio, f_value):
        aji = Aji(lambda0, g_ratio, f_value)
        bji = Bji(aji, lambda0)
        bji_wavelength = Bji_lambda(aji, lambda0)
        bij = Bij(bji, g_ratio)
        bij_wavelength = Bij(bji_wavelength, g_ratio)
        return cls(
            Aji=aji,
            Bji=bji,
            Bji_wavelength=bji_wavelength,
            Bij=bij,
            Bij_wavelength=bij_wavelength,
        )

def constant_stark_linear_sutton(n_upper: int, n_lower: int):
    """
    Constant for hydrogenic linear Stark broadening following the implementation
    of Sutton (1978).
    Apply as constant * n_e**(2/3).

    Result in m3 rad / s

    Parameters
    ----------
    n_upper : int
        Principal quantum number of the upper level of the transition
    n_lower : int
        Principal quantum number of the lower level of the transition
    """
    # NOTE(cmo): Different implementation to Lightweaver, which may be
    # missing 2pi term. Based on Tiago's work/RH1.5d. Appears correct as per
    # Sutton paper.

    a1 = 0.642 if n_upper - n_lower == 1 else 1.0
    c = (
        4.0
        * np.pi
        * 0.425
        * a1
        * 0.6e-4
        * (n_upper**2 - n_lower**2)
        * (u.m**3 * u.rad / u.s)
    )
    return c

def c4_traving(upper_energy: u.Quantity[u.J], lower_energy: u.Quantity[u.J], overlying_cont_energy: u.Quantity[u.J], stage: int):
    """
    Computes the C4 constant for quadratic Stark broadening following the
    Traving (1960) formalism.

    Result in m4 / s

    Parameters
    ----------
    upper_energy : float (J)
        The energy of the upper level of the transition.
    lower_energy : float (J)
        The energy of the lower level of the transition.
    overlying_cont_energy : float (J)
        The energy of the next overlying continuum level.
    stage : int
        The effective ionisation stage of the specie.
    """
    n_eff_u = n_eff(overlying_cont_energy, upper_energy, stage)
    n_eff_l = n_eff(overlying_cont_energy, lower_energy, stage)
    Z = stage

    C4 = (
        const.e.si**2
        / (4.0 * np.pi * const.eps0)
        * const.a0**3
        * 2.0
        * np.pi
        / (const.h * 18 * Z**4)
        * (
            (n_eff_u * (5.0 * n_eff_u**2 + 1.0)) ** 2
            - (n_eff_l * (5.0 * n_eff_l**2 + 1.0)) ** 2
        )
    )
    C4 <<= u.m**4 / u.s
    return C4

def constant_stark_quadratic(
    upper_energy: u.Quantity[u.J],
    lower_energy: u.Quantity[u.J],
    overlying_cont_energy: u.Quantity[u.J],
    stage: int,
    mass: u.Quantity[u.u],
    mean_atomic_mass: Optional[u.Quantity[u.u]] = None,
    scaling: float = 1.0
):
    """
    Constant for quadratic Stark broadening using the RH recipe and following the approach of Traving (1960) for C4.

    Apply as constant * T**(1/6) * n_e

    Parameters
    ----------
    upper_energy : float (J)
        The energy of the upper level of the transition.
    lower_energy : float (J)
        The energy of the lower level of the transition.
    overlying_cont_energy : float (J)
        The energy of the next overlying continuum level.
    stage : int
        The effective ionisation stage of the specie.
    mass : float (amu)
        The mass of the element
    mean_atomic_mass : float (amu), optional
        The abundance weighted mean atomic mass (in amu). Will be drawn from
        Asplund et al (2009) if not provided.
    scaling : float
        An additional scaling term
    """

    if mean_atomic_mass is None:
        mean_atomic_mass = lw.DefaultAtomicAbundance.avgMass << u.u

    v_rel_const = (8.0 * const.k_B / (np.pi * mass) << u.Unit("J / (K kg)")).value
    v_rel_e = (1.0 + mass / const.m_e)**(1.0 / 6.0)
    v_rel_s = (1.0 + mass / mean_atomic_mass)**(1.0 / 6.0)
    v_rel_term = v_rel_const**(1.0 / 6.0) * (v_rel_e + v_rel_s)
    C4 = c4_traving(upper_energy, lower_energy, overlying_cont_energy, stage)
    C4_23 = (scaling * C4).value**(2.0 / 3.0)
    return (11.37 * u.Unit("m3 rad / s")) * v_rel_term * C4_23
