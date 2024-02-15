import numpy as np
import pytest
import astropy.units as u
import astropy.constants as const
import lightweaver as lw
from crtaf.physics_utils import (
    EinsteinCoeffs,
    c4_traving,
    constant_stark_linear_sutton,
    constant_stark_quadratic,
    constant_unsold,
)
from lightweaver.rh_atoms import CaII_atom, H_6_atom
from lightweaver.broadening import (
    QuadraticStarkBroadening as LwQuadraticStarkBroadening,
    HydrogenLinearStarkBroadening as LwHydrogenLinearStarkBroadening,
    VdwUnsold as LwVdwUnsold,
)


def test_sutton():
    stark = constant_stark_linear_sutton(3, 1)
    assert stark.value == pytest.approx(0.0025635396)
    assert stark.unit == u.m**3 * u.rad / u.s

    stark_32 = constant_stark_linear_sutton(3, 2)
    assert stark_32.value == pytest.approx(0.00102862026)

    h = H_6_atom()
    l = h.lines[4]
    b = l.broadening.elastic[-1]
    assert isinstance(b, LwHydrogenLinearStarkBroadening)
    atmos = lw.Atmosphere.make_1d(
        scale=lw.ScaleType.Geometric,
        depthScale=np.array([1.0, 0.0]),
        temperature=np.array([1e5, 1e5]),
        vlos=np.zeros(2),
        vturb=np.ones(2),
        ne=np.ones(2),
        nHTot=np.ones(2),
    )
    # TODO(cmo): This depends on Lightweaver continuing to follow the old, likely incorrect RH interpretation.
    assert stark_32.value == pytest.approx(
        4.0 * np.pi * 0.425 * b.broaden(atmos, None)[0], abs=0.0, rel=1e-4
    )


def test_quadratic_stark():
    c4 = c4_traving(1.1 * u.aJ, 1.0 * u.aJ, 1.5 * u.aJ, 1)
    assert c4.value == pytest.approx(3.74474161e-23, abs=0.0, rel=1e-6)
    c4_2 = c4_traving(6 * u.eV, 5 * u.eV, 8 * u.eV, 1)
    assert c4_2.value == pytest.approx(1.04254924175e-22, abs=0.0, rel=1e-6)
    # NOTE(cmo): Matching Tiago's 1kg test
    cst = constant_stark_quadratic(1.1 * u.aJ, 1.0 * u.aJ, 1.5 * u.aJ, 1, (1.0 << u.kg))
    assert cst.value == pytest.approx(2.72658480282e-13, abs=0.0, rel=1e-6)

    # NOTE(cmo): Compare to Lw
    ca = CaII_atom()
    l = ca.lines[0]
    b = l.broadening.elastic[1]
    assert isinstance(b, LwQuadraticStarkBroadening)
    lw_val = b.cStark23 * b.C ** (1.0 / 6.0) * b.Cm
    cst_ca = constant_stark_quadratic(
        l.jLevel.E_eV << u.eV,
        l.iLevel.E_eV << u.eV,
        l.overlyingContinuumLevel.E_eV << u.eV,
        l.iLevel.stage + 1,
        ca.element.mass << u.u,
    )
    assert cst_ca.value == pytest.approx(lw_val, abs=0.0, rel=1e-4)
    assert cst_ca.unit == u.Unit("m3 rad / s")


def test_vdw_unsold():
    cst = constant_unsold(1.1 * u.aJ, 1.0 * u.aJ, 1.5 * u.aJ, 1, 1.0 * u.kg)
    assert cst.unit == u.m**3 * u.rad / u.s
    # NOTE(cmo): This number different to Transparency.jl because assumption of different He abundance
    assert cst.value == pytest.approx(1.104490946e-15, abs=0.0, rel=1e-6)
    h = H_6_atom()
    l = h.lines[0]
    b = l.broadening.elastic[0]
    assert isinstance(b, LwVdwUnsold)
    cst_h = constant_unsold(
        l.jLevel.E_eV << u.eV,
        l.iLevel.E_eV << u.eV,
        l.overlyingContinuumLevel.E_eV << u.eV,
        l.iLevel.stage + 1,
        h.element.mass << u.u,
    )
    he_abund = lw.DefaultAtomicAbundance["He"]
    cst_lw = 8.08 * (b.vRel35H + he_abund * b.vRel35He) * b.C625
    assert cst_h.value == pytest.approx(cst_lw, abs=0.0, rel=1e-4)


def test_einstein_coeffs():
    h = H_6_atom()
    l = h.lines[4]
    ec = EinsteinCoeffs.compute(l.lambda0 << u.nm, l.iLevel.g / l.jLevel.g, l.f)

    assert ec.Aji.value == pytest.approx(l.Aji)
    assert ec.Aji.unit == u.Unit("1 / s")
    assert ec.Bji.value == pytest.approx(l.Bji)
    assert ec.Bji.unit == u.Unit("m2 / (J s)")
    assert ec.Bij.value == pytest.approx(l.Bij)
    assert ec.Bji_wavelength.value == pytest.approx(4.5115938068e-8, abs=0.0, rel=1e-6)
    assert ec.Bij_wavelength.value == pytest.approx(1.01510860e-7, abs=0.0, rel=1e-6)
    assert ec.Bij_wavelength.unit == u.Unit("m3 / J")
