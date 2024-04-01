from typing import TYPE_CHECKING, Any, List, Optional

import astropy.constants as const
import astropy.units as u
import lightweaver as lw
import numpy as np
from crtaf.core_types import (
    Atom,
    AtomicBoundBound,
    AtomicLevel,
    HydrogenicBoundFree,
    LinearGrid,
    NaturalBroadening,
    ScaledExponents,
    StarkLinearSutton,
    StarkMultiplicative,
    StarkQuadratic,
    TabulatedBoundFree,
    TabulatedGrid,
    TemperatureInterpolationRateImpl,
    TransCollisionalRates,
    VdWUnsold,
    VoigtBoundBound,
)
from crtaf.exts.linear_core_exp_wings_grid import (
    LinearCoreExpWings,
    simplify_linear_core_exp_wings,
)
from crtaf.exts.multi_wavelength_grid import (
    MultiWavelengthGrid,
    simplify_multi_wavelength_grid,
)

from crtaf.physics_utils import (
    EinsteinCoeffs,
    compute_lambda0,
    constant_stark_linear_sutton,
    constant_stark_quadratic,
    constant_unsold,
    gaunt_bf,
    n_eff,
)


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
        g_upper = atom.levels[line.transition[0]].g
        g_lower = atom.levels[line.transition[1]].g
        n_upper = int(np.round(np.sqrt(0.5 * g_upper)))
        n_lower = int(np.round(np.sqrt(0.5 * g_lower)))
    else:
        n_upper = b.n_upper
        n_lower = b.n_lower

    c = constant_stark_linear_sutton(n_upper, n_lower)
    return ScaledExponents(
        type="Scaled_Exponents",
        elastic=b.elastic,  # type: ignore
        scaling=c.to(u.m**3 / u.s, equivalencies=u.dimensionless_angles()).value,
        temperature_exponent=0.0,
        hydrogen_exponent=0.0,
        electron_exponent=2.0 / 3.0,
    )


def get_overlying_continuum(atom: Atom, level: AtomicLevel):
    """
    Returns the level name and level with the lowest energy at the next ionisation stage.
    """
    stage = level.stage
    next_stage_energies = [
        (name, l.energy) for name, l in atom.levels.items() if l.stage == stage + 1
    ]
    overlying_cont_name = min(next_stage_energies, key=lambda x: x[1])[0]  # type: ignore
    return overlying_cont_name, atom.levels[overlying_cont_name]


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
    _, overlying_cont = get_overlying_continuum(atom, atom.levels[line.transition[1]])
    overlying_cont_energy = overlying_cont.energy

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
        scaling=b.scaling,  # type: ignore
    )

    return ScaledExponents(
        type="Scaled_Exponents",
        elastic=b.elastic,  # type: ignore
        scaling=b.scaling
        * cst.to(u.m**3 / u.s, equivalencies=u.dimensionless_angles()).value,
        temperature_exponent=(1.0 / 6.0),
        hydrogen_exponent=0.0,
        electron_exponent=1.0,
    )


def simplify_stark_multiplicative(b: StarkMultiplicative, *args, **kwargs):
    return ScaledExponents(
        type="Scaled_Exponents",
        elastic=b.elastic,  # type: ignore
        scaling=b.scaling * b.C_4.to(u.m**3 / u.s).value,
        temperature_exponent=0.0,
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

    stage = atom.levels[line.transition[0]].stage
    e_j = atom.levels[line.transition[0]].energy
    e_i = atom.levels[line.transition[1]].energy
    _, overlying_cont = get_overlying_continuum(atom, atom.levels[line.transition[1]])
    e_cont = overlying_cont.energy

    coeff = constant_unsold(
        e_j, e_i, e_cont, stage, mass << u.u, b.H_scaling, b.He_scaling
    )

    return ScaledExponents(
        type="Scaled_Exponents",
        elastic=b.elastic,  # type: ignore
        scaling=coeff.value,
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


def simplify_voigt_line(
    l: VoigtBoundBound, roots: Optional[List[Any]], visitor, **kwargs
):
    if roots is None:
        raise ValueError("roots must be provided (call via visitor interface).")

    atom = roots[0]
    new_roots = roots + [l]
    trans = l.transition

    broadening = [visitor.visit(b, roots=new_roots) for b in l.broadening]
    wavelength_grid = visitor.visit(l.wavelength_grid, roots=new_roots)

    lambda0 = compute_lambda0(atom, l)
    lambda0_m = lambda0.to(u.m)
    g_ij = atom.levels[trans[1]].g / atom.levels[trans[0]].g

    coeffs = EinsteinCoeffs.compute(lambda0_m, g_ij, l.f_value)

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
        lambda0=lambda0,
    )


def simplify_hydrogenic_cont(
    c: HydrogenicBoundFree, roots: Optional[List[Any]], *args, **kwargs
):

    if roots is None:
        raise ValueError("roots must be provided (call via visitor interface).")

    atom = roots[0]
    lambda_edge = compute_lambda0(atom, c)
    if c.lambda_min >= lambda_edge:
        raise ValueError("Min lambda for a continuum can't be bigger than its edge!")

    wavelengths = (
        np.linspace(
            c.lambda_min.to(u.nm).value,
            # NOTE(cmo): We pull back from the edge very slightly to avoid codes like Lw potentially chopping it off.
            lambda_edge.to(u.nm).value - 1e-8,
            c.n_lambda,
        )
        << u.nm
    )
    z_eff = atom.levels[c.transition[0]].stage - 1
    e_upper = atom.levels[c.transition[0]].energy.to(u.J, equivalencies=u.spectral())
    e_lower = atom.levels[c.transition[1]].energy.to(u.J, equivalencies=u.spectral())
    n_effective = n_eff(e_upper, e_lower, z_eff)
    g_bf0 = gaunt_bf(lambda_edge, z_eff, n_effective)
    g_bf = gaunt_bf(wavelengths, z_eff, n_effective)
    sigma = c.sigma_peak.to(u.m**2) * g_bf / g_bf0 * (wavelengths / lambda_edge) ** 3

    return TabulatedBoundFree(
        type="Tabulated",
        transition=c.transition,
        wavelengths=wavelengths.to(u.nm),
        sigma=sigma.to(u.m**2),
    )


def simplify_tabulated_cont(c: TabulatedBoundFree, *args, **kwargs):
    return c.simplify()


def simplify_temperature_interp_rate(
    r: TemperatureInterpolationRateImpl, *args, **kwargs
):
    return r.simplify()


def simplify_trans_coll_rate(
    r: TransCollisionalRates, roots: Optional[List[Any]], visitor, *args, **kwargs
):
    if roots is None:
        roots = [r]
    else:
        roots = roots + [r]

    simplified_rates = [visitor.visit(rate) for rate in r.data]
    return TransCollisionalRates(
        transition=r.transition,
        data=simplified_rates,
    )


def default_visitors():
    visitors = {
        AtomicLevel: simplify_atomic_level,
        NaturalBroadening: simplify_natural_broadening,
        StarkLinearSutton: simplify_stark_linear,
        StarkQuadratic: simplify_stark_quadratic,
        StarkMultiplicative: simplify_stark_multiplicative,
        VdWUnsold: simplify_vdw_unsold,
        ScaledExponents: simplify_scaled_exponents,
        LinearGrid: simplify_linear_grid,
        TabulatedGrid: simplify_tabulated_grid,
        VoigtBoundBound: simplify_voigt_line,
        HydrogenicBoundFree: simplify_hydrogenic_cont,
        TabulatedBoundFree: simplify_tabulated_cont,
        TemperatureInterpolationRateImpl: simplify_temperature_interp_rate,
        TransCollisionalRates: simplify_trans_coll_rate,
        # core exts
        LinearCoreExpWings: simplify_linear_core_exp_wings,
        MultiWavelengthGrid: simplify_multi_wavelength_grid,
    }
    return visitors
