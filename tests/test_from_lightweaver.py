from crtaf.from_lightweaver import LightweaverAtomConverter
from crtaf.core_types import PrdVoigtBoundBound, VoigtBoundBound
from lightweaver.rh_atoms import H_6_atom, CaII_atom


def test_H_conversion():
    h = H_6_atom()
    conv = LightweaverAtomConverter()
    crtaf_h = conv.convert(h)

    assert isinstance(crtaf_h.radiative_bound_bound[0], PrdVoigtBoundBound)
    assert isinstance(crtaf_h.radiative_bound_bound[4], VoigtBoundBound)


def test_CaII_conversion():
    ca = CaII_atom()
    conv = LightweaverAtomConverter()
    crtaf_ca = conv.convert(ca)

    assert isinstance(crtaf_ca.radiative_bound_bound[0], PrdVoigtBoundBound)
    assert isinstance(crtaf_ca.radiative_bound_bound[4], VoigtBoundBound)
