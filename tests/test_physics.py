import pytest
import astropy.units as u
import astropy.constants as const
from crtaf.physics_utils import c4_traving, constant_stark_linear_sutton, constant_stark_quadratic
from lightweaver.rh_atoms import CaII_atom
from lightweaver.broadening import QuadraticStarkBroadening as LwQuadraticStarkBroadening

def test_sutton():
    stark = constant_stark_linear_sutton(3, 1)
    assert stark.value == pytest.approx(0.0025635396)
    assert stark.unit == u.m**3 * u.rad / u.s

    stark_32 = constant_stark_linear_sutton(3, 2)
    assert stark_32.value == pytest.approx(0.00102862026)

def test_quadratic_stark():
    c4 = c4_traving(1.1 * u.aJ, 1.0 * u.aJ, 1.5 * u.aJ, 1)
    assert c4.value == pytest.approx(3.74474161e-23)
    c4_2 = c4_traving(6 * u.eV, 5 * u.eV, 8 * u.eV, 1)
    assert c4_2.value == pytest.approx(1.04254924175e-22)
    # NOTE(cmo): Matching Tiago's 1kg test
    cst = constant_stark_quadratic(1.1 * u.aJ, 1.0 * u.aJ, 1.5 * u.aJ, 1, ((1.0/const.u.value) << u.kg))
    assert cst.value == pytest.approx(2.983753e-13)

    # NOTE(cmo): Compare to Lw
    ca = CaII_atom()
    l = ca.lines[0]
    b = l.broadening.elastic[1]
    assert isinstance(b, LwQuadraticStarkBroadening)
    lw_val = b.cStark23 * b.C**(1.0 / 6.0) * b.Cm
    cst_ca = constant_stark_quadratic(
        l.jLevel.E_eV << u.eV,
        l.iLevel.E_eV << u.eV,
        l.overlyingContinuumLevel.E_eV << u.eV,
        l.iLevel.stage,
        ca.element.mass << u.u
    )
    assert cst_ca.value == pytest.approx(lw_val)
    assert cst_ca.unit == u.Unit("m3 rad / s")