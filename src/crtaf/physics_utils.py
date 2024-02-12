from dataclasses import dataclass
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