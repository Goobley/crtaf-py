from typing import TYPE_CHECKING, Any, List, Optional

import astropy.constants as const
import astropy.units as u
import lightweaver as lw
import numpy as np
from crtaf.core_types import Atom, AtomicBoundBound, AtomicLevel, HydrogenicBoundFree, LinearGrid, NaturalBroadening, ScaledExponents, StarkLinearSutton, StarkQuadratic, TabulatedBoundFree, TabulatedGrid, TransCollisionalRates, VdWUnsold, VoigtBoundBound

from crtaf.physics_utils import EinsteinCoeffs, constant_stark_linear_sutton, constant_stark_quadratic, n_eff

def simplify_atomic_level(level: AtomicLevel, *args, **kwargs):
    return level.simplify()


def simplify_natural_broadening(b: NaturalBroadening, *args, **kwargs):
    return b.simplify()


def simplify_stark_linear(
    b: StarkLinearSutton, roots: Optional[List[Any]], *args, **kwargs
):
    if roots is None:
        raise ValueError("roots must be provided (call via visitor interface).")
    atom = roots[0]
    line = roots[-1]

    if b.n_upper is None or b.n_lower is None:
        g_upper = atom.levels[line.transition[0]]
        g_lower = atom.levels[line.transition[1]]
        n_upper = int(np.round(np.sqrt(0.5 * g_upper)))
        n_lower = int(np.round(np.sqrt(0.5 * g_lower)))
    else:
        n_upper = b.n_upper
        n_lower = b.n_lower

    c = constant_stark_linear_sutton(n_upper, n_lower)
    return ScaledExponents(
        type="Scaled_Exponents",
        scaling=c,
        temperature_exponent=0.0,
        hydrogen_exponent=0.0,
        electron_exponent=2 / 3,
    )


def simplify_stark_quadratic(
    b: StarkQuadratic, roots: Optional[List[Any]], *args, **kwargs
):
    if roots is None:
        raise ValueError("roots must be provided (call via visitor interface).")

    atom: Atom = roots[0]
    line: AtomicBoundBound = roots[-1]
    e_j = atom.levels[line.transition[0]].energy
    e_i = atom.levels[line.transition[1]].energy
    stage = atom.levels[line.transition[0]].stage
    next_stage_energies = [l.energy for l in atom.levels if l.stage == stage + 1]
    overlying_cont_energy = min(next_stage_energies)

    if b.C_4 is None:
        if atom.element.atomic_mass is not None:
            mass = atom.element.atomic_mass << u.u
        else:
            mass = lw.PeriodicTable[atom.element.symbol].mass << u.u
        cst = constant_stark_quadratic(
            e_j,
            e_i,
            overlying_cont_energy,
            stage,
            mass,
            b.scaling,
        )
    else:
        cst = b.C_4
        temperature_exponent = 0.0

    return ScaledExponents(
        type="Scaled_Exponents",
        scaling=b.scaling * cst,
        temperature_exponent=temperature_exponent,
        hydrogen_exponent=0.0,
        electron_exponent=1.0,
    )


def simplify_vdw_unsold(b: VdWUnsold, roots: Optional[List[Any]], *args, **kwargs):
    if roots is None:
        raise ValueError("roots must be provided (call via visitor interface).")

    atom = roots[0]
    line = roots[-1]
    mass = atom.element.atomic_mass
    if mass is None:
        mass = lw.PeriodicTable[atom.element.symbol].mass

    Z = atom.levels[line.transition[0]].stage + 1
    e_j = atom.levels[line.transition[0]].energy
    e_i = atom.levels[line.transition[1]].energy
    cont = min([l for l in atom.levels if l.stage == Z], key=lambda x: x.energy)
    e_cont = cont.energy
    abar_h = 4.5 * 4.0 * np.pi * const.eps0 * const.a0**3  # polarizability of H [Fm^2]


    ryd = const.Ryd.to(u.J, equivalencies=u.spectral())
    deltaR = ((ryd / (e_cont - e_j)) ** 2 - (ryd / (e_cont - e_i)) ** 2).to(u.J / u.J)
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
        * deltaR
    ) ** 0.4

    v_rel_const = 8.0 * const.k_B / np.pi * const.u
    v_rel_h = v_rel_const * (1.0 + mass / lw.PeriodicTable['H'].mass)
    v_rel_he = v_rel_const * (1.0 + mass / lw.PeriodicTable['He'].mass)
    coeff = 8.08 * (b.H_scaling * v_rel_h**0.3 + b.He_scaling * v_rel_he**0.3) * C6

    return ScaledExponents(
        type="Scaled_Exponents",
        scaling=coeff,
        temperature_exponent=0.3,
        hydrogen_exponent=1.0,
        electron_exponent=0.0,
    )

def simplify_scaled_exponents(b: ScaledExponents, *args, **kwargs):
    return b.simplify()

def simplify_linear_grid(g: LinearGrid, *args, **kwargs):
    return g.simplify()

def simplify_tabulated_grid(g: TabulatedGrid, *args, **kwargs):
    return g.simplify()

def simplify_voigt_line(l: VoigtBoundBound, roots: Optional[List[Any]], visitor, **kwargs):
    if roots is None:
        raise ValueError("roots must be provided (call via visitor interface).")

    atom = roots[0]
    new_roots = roots + [l]
    trans = l.transition

    broadening = [visitor.visit(b, roots=new_roots) for b in l.broadening]
    wavelength_grid = visitor.visit(l.wavelength_grid, roots=new_roots)

    delta_E = (atom.levels[trans[0]].energy - atom.levels[trans[1]].energy).to(u.J, equivalencies=u.spectral())
    lambda0 = ((const.h * const.c) / (delta_E)).to(u.nm)
    lambda0_m = lambda0.to(u.m)
    g_ratio = atom.levels[trans[0]].g / atom.levels[trans[1]].g

    coeffs = EinsteinCoeffs.compute(lambda0_m, g_ratio, l.f_value)

    return l.__class__(
        type=l.type,
        transition=l.transition,
        f_value=l.f_value,
        broadening=broadening,
        wavelength_grid=wavelength_grid,
        Aji=coeffs.Aji,
        Bji=coeffs.Bji,
        Bji_wavelength=coeffs.Bji_wavelength,
        Bij=coeffs.Bij,
        Bij_wavelength=coeffs.Bij_wavelength,
        lambda0=lambda0
    )

def simplify_hydrogenic_cont(c: HydrogenicBoundFree, *args, **kwargs):
    return c.simplify()

def simplify_tabulated_cont(c: TabulatedBoundFree, *args, **kwargs):
    return c.simplify()

def simplify_trans_coll_rate(c: TransCollisionalRates, *args, **kwargs):
    return c.simplify()

def default_visitors():
    visitors = {
        AtomicLevel: simplify_atomic_level,
        NaturalBroadening: simplify_natural_broadening,
        StarkLinearSutton: simplify_stark_linear,
        StarkQuadratic: simplify_stark_quadratic,
        VdWUnsold: simplify_vdw_unsold,
        ScaledExponents: simplify_scaled_exponents,
        LinearGrid: simplify_linear_grid,
        TabulatedGrid: simplify_tabulated_grid,
        VoigtBoundBound: simplify_voigt_line,
        HydrogenicBoundFree: simplify_hydrogenic_cont,
        TabulatedBoundFree: simplify_tabulated_cont,
        TransCollisionalRates: simplify_trans_coll_rate,
    }
    return visitors
