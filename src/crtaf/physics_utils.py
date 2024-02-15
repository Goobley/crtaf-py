from dataclasses import dataclass
from typing import Optional
import astropy.constants as const
import astropy.units as u
import lightweaver as lw
import numpy as np
from lightweaver.barklem import Barklem


def n_eff(
    e_upper: u.Quantity[u.J],
    e_lower: u.Quantity[u.J],
    Z: int,
    mass: Optional[u.Quantity[u.u]] = None,
):
    """
    Effective principal quantum number for transition

    Parameters
    ----------
    e_upper : float (J)
        The energy of the upper level of the transition
    e_lower : float (J)
        The energy of the lower level of the transition
    Z : int
        The effective nuclear charge of the specie (1 is neutral).
    mass : float (u), optional
        The mass of the element in amu. If not provided will be computed as if
        it were H.
    """
    e_rydberg = const.Ryd.to(u.J, equivalencies=u.spectral())
    if mass is None:
        h_amu = lw.PeriodicTable["H"].mass
        mass = h_amu << u.u
    # NOTE: H ionisation
    ry_h = e_rydberg / (1.0 + const.m_e / mass)

    e_u = e_upper.to(u.J)
    e_l = e_lower.to(u.J)
    return Z * np.sqrt((ry_h / (e_u - e_l)).to(u.J / u.J))


def Aji(lambda0, g_ij, f_value):
    cst = 2.0 * np.pi * (const.e.si / const.eps0) * (const.e.si / const.m_e) / const.c
    return (cst / lambda0**2 * g_ij * f_value).to(1.0 / u.s)


def Bji(Aji, lambda0):
    result = Aji * lambda0**3 / (2.0 * const.h * const.c)
    return result.to("m2 / (J s)")


def Bji_lambda(Aji, lambda0):
    result = Aji * lambda0**5 / (2.0 * const.h * const.c**2)
    return result.to("m3 / J")


def Bij(Bji, g_ij):
    return Bji / g_ij


@dataclass
class EinsteinCoeffs:
    Aji: u.Quantity
    Bji: u.Quantity
    Bji_wavelength: u.Quantity
    Bij: u.Quantity
    Bij_wavelength: u.Quantity

    @classmethod
    def compute(cls, lambda0: u.Quantity[u.m], g_ij: float, f_value: float):
        """
        Computes the Einstein coefficients for a bound-bound transition.

        Parameters
        ----------
        lambda0 : float (m)
            The rest wavelength of the transition.
        g_ij : float
            g_i / g_j for the transition.
        f_value : float
            Oscillator strength for the transition.
        """
        aji = Aji(lambda0, g_ij, f_value)
        bji = Bji(aji, lambda0)
        bji_wavelength = Bji_lambda(aji, lambda0)
        bij = Bij(bji, g_ij)
        bij_wavelength = Bij(bji_wavelength, g_ij)
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


def c4_traving(
    upper_energy: u.Quantity[u.J],
    lower_energy: u.Quantity[u.J],
    overlying_cont_energy: u.Quantity[u.J],
    stage: int,
    mass: Optional[u.Quantity[u.u]] = None,
):
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
        The effective nuclear charge of the specie (1 is neutral).
    mass : float (u), optional
        The mass of the species in amu, assumed to be H if not provided.
    """
    upper_energy = upper_energy.to(u.J, equivalencies=u.spectral())
    lower_energy = lower_energy.to(u.J, equivalencies=u.spectral())
    overlying_cont_energy = overlying_cont_energy.to(u.J, equivalencies=u.spectral())
    n_eff_u = n_eff(overlying_cont_energy, upper_energy, stage, mass=mass)
    n_eff_l = n_eff(overlying_cont_energy, lower_energy, stage, mass=mass)
    Z = stage

    C4 = (
        const.e.si**2
        / (4.0 * np.pi * const.eps0)
        * const.a0
        * (2.0 * np.pi * const.a0**2 / const.h)
        / (18.0 * Z**4)
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
    scaling: float = 1.0,
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
        Mean atomic mass (in amu). Will be taken as 28.0 if not provided.
    scaling : float
        An additional scaling term for C4
    """

    if mean_atomic_mass is None:
        mean_atomic_mass = 28.0 << u.u

    v_rel_const = (8.0 * const.k_B / (np.pi * mass) << u.Unit("J / (K kg)")).value
    v_rel_e = (1.0 + (mass / const.m_e)) ** (1.0 / 6.0)
    v_rel_s = (1.0 + (mass / mean_atomic_mass)) ** (1.0 / 6.0)
    v_rel_term = v_rel_const ** (1.0 / 6.0) * (v_rel_e + v_rel_s)
    C4 = c4_traving(upper_energy, lower_energy, overlying_cont_energy, stage, mass=mass)
    C4_23 = (scaling * C4).value ** (2.0 / 3.0)
    return (11.37 * u.Unit("m3 rad / s")) * v_rel_term * C4_23


def constant_unsold(
    upper_energy: u.Quantity[u.J],
    lower_energy: u.Quantity[u.J],
    overlying_cont_energy: u.Quantity[u.J],
    stage: int,
    mass: u.Quantity[u.u],
    H_scaling: float = 1.0,
    He_scaling: float = 1.0,
):
    """
    Constant for Van der Waals broadening following Unsold, based on Mihalas description.
    He abundance as per Asplund 2009.
    Apply as constant * T**0.3 * n_H

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
    H_scaling : float, optional
        An additional scaling term for interactions with H
    He_scaling : float, optional
        An additional scaling term for interactions with He
    """

    e_j = upper_energy.to(u.J, equivalencies=u.spectral())
    e_i = lower_energy.to(u.J, equivalencies=u.spectral())
    e_cont = overlying_cont_energy.to(u.J, equivalencies=u.spectral())
    Z = stage
    abar_h = (
        4.5 * 4.0 * np.pi * const.eps0 * const.a0**3
    )  # polarizability of H [Fm^2]

    ryd = const.Ryd.to(u.J, equivalencies=u.spectral())
    delta_r = ((ryd / (e_cont - e_j)) ** 2 - (ryd / (e_cont - e_i)) ** 2).to(u.J / u.J)
    inv_4pieps0 = 1.0 / (4.0 * np.pi * const.eps0)
    C6 = (
        2.5
        * const.e.si**2
        * inv_4pieps0
        * abar_h
        * inv_4pieps0
        * 2.0
        * np.pi
        * (Z * const.a0) ** 2
        / const.h
        * delta_r
    ).to(u.Unit("C2 m6 / (F J s)"))

    v_rel_const = (8.0 * const.k_B / (np.pi * mass)).to(u.Unit("J / (K kg)")).value
    v_rel_h = v_rel_const * (1.0 + mass / (lw.PeriodicTable["H"].mass << u.u))
    v_rel_he = v_rel_const * (1.0 + mass / (lw.PeriodicTable["He"].mass << u.u))
    he_abund = lw.DefaultAtomicAbundance["He"]
    coeff = (
        8.08
        * (H_scaling * v_rel_h**0.3 + He_scaling * he_abund * v_rel_he**0.3)
        * C6.value**0.4
        * u.Unit("m3 rad / s")
    )
    return coeff
