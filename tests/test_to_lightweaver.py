# NOTE(cmo): Using the harness here to test the new crtaf behaviour added to Lw

from copy import deepcopy
from matplotlib import pyplot as plt
import pytest
import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom, CaII_atom
from lightweaver.fal import Falc82

GOT_CRTAF = True
try:
    from lightweaver.crtaf import from_crtaf
except ImportError:
    GOT_CRTAF = False
import numpy.testing as npt

from crtaf.from_lightweaver import LightweaverAtomConverter
from crtaf import default_visitors, AtomicSimplificationVisitor


def falc_model(h_model=None, ca_model=None, active_atoms=None):
    if h_model is None:
        h_model = H_6_atom()
    if ca_model is None:
        ca_model = CaII_atom()
    if active_atoms is None:
        active_atoms = ["Ca"]

    atmos = Falc82()
    atmos.quadrature(3)
    a_set = lw.RadiativeSet([h_model, ca_model])
    a_set.set_active(*active_atoms)

    spect = a_set.compute_wavelength_grid()
    eq_pops = a_set.compute_eq_pops(atmos)
    ctx = lw.Context(atmos, spect, eq_pops)
    lw.iterate_ctx_se(ctx, quiet=True)
    I = ctx.spect.I[:, -1]
    return ctx.spect.wavelength, I


@pytest.mark.skipif(
    not GOT_CRTAF,
    reason="A version of lightweaver with CRTAF integration is not available",
)
def test_lw_caii():
    ca_wave_base, baseline = falc_model(ca_model=CaII_atom())

    ca_crtaf = LightweaverAtomConverter().convert(CaII_atom())
    ca_roundtrip = from_crtaf(ca_crtaf)

    ca_wave_roundtrip, roundtrip = falc_model(ca_model=ca_roundtrip)
    npt.assert_allclose(ca_wave_base, ca_wave_roundtrip)
    npt.assert_allclose(roundtrip, baseline, atol=0.0)

    visitor = AtomicSimplificationVisitor(default_visitors())
    ca_crtaf_simplified = ca_crtaf.simplify_visit(visitor)
    ca_crtaf_simplified_roundtrip = from_crtaf(ca_crtaf_simplified)

    ca_wave_roundtrip_2, roundtrip_2 = falc_model(
        ca_model=ca_crtaf_simplified_roundtrip, active_atoms=["Ca"]
    )
    npt.assert_allclose(ca_wave_base, ca_wave_roundtrip_2)
    # NOTE(cmo): Less perfect match due to ScaledExponents
    npt.assert_allclose(roundtrip_2, baseline, atol=0.0, rtol=3e-5)


@pytest.mark.skipif(
    not GOT_CRTAF,
    reason="A version of lightweaver with CRTAF integration is not available",
)
def test_lw_H6():
    h_wave_baseline, baseline_H = falc_model(h_model=H_6_atom(), active_atoms=["H"])

    h_crtaf = LightweaverAtomConverter().convert(H_6_atom())
    h_roundtrip = from_crtaf(h_crtaf)
    h_wave_roundtrip, roundtrip_H = falc_model(h_model=h_roundtrip, active_atoms=["H"])
    npt.assert_allclose(h_wave_baseline, h_wave_roundtrip)
    npt.assert_allclose(roundtrip_H, baseline_H, atol=0.0)

    visitor = AtomicSimplificationVisitor(default_visitors())
    h_crtaf_simplified = h_crtaf.simplify_visit(visitor)
    h_crtaf_simplified_roundtrip = from_crtaf(h_crtaf_simplified)
    h_wave_roundtrip_2, roundtrip_H_simplified = falc_model(
        h_model=h_crtaf_simplified_roundtrip, active_atoms=["H"]
    )
    npt.assert_allclose(h_wave_baseline, h_wave_roundtrip_2)

    with pytest.raises(AssertionError):
        # NOTE(cmo): This is expected to fail doe to the difference (significant
        # in Linear Stark broadening (a factor of 5!))
        npt.assert_allclose(roundtrip_H_simplified, baseline_H, atol=0.0)

    # NOTE(cmo): Test the continuum conversion (using the non-simplified lines)
    h_crtaf_simplified_mix = deepcopy(h_crtaf_simplified)
    h_crtaf_simplified_mix.lines = h_crtaf.lines
    h_mix_lw = from_crtaf(h_crtaf_simplified_mix)
    h_wave_roundtrip_3, roundtrip_H_mix = falc_model(
        h_model=h_mix_lw, active_atoms=["H"]
    )
    npt.assert_allclose(h_wave_baseline, h_wave_roundtrip_3)
    npt.assert_allclose(roundtrip_H_mix, baseline_H, atol=0.0, rtol=2e-4)
