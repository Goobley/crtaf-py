from typing import Any, List, Literal, Optional
from typing_extensions import Annotated
from annotated_types import Ge
import numpy as np
from pydantic import field_validator
from crtaf import WavelengthGrid, AstropyQty, TabulatedGrid
import astropy.units as u
import astropy.constants as const

from crtaf.physics_utils import compute_lambda0


class MultiWavelengthGrid(WavelengthGrid, type_name="MULTI"):
    _crtaf_ext_name: str = "multi_wavelength_grid"

    type: str = Literal["MULTI"]
    q_max: Annotated[float, Ge(0.0)]
    q0: Annotated[float, Ge(0.0)]
    n_lambda: int
    q_norm: AstropyQty

    @field_validator("q_norm")
    @classmethod
    def _validate(cls, v):
        v.to("m / s")
        return v


def simplify_multi_wavelength_grid(
    quad: MultiWavelengthGrid, roots: Optional[List[Any]], visitor, *args, **kwargs
) -> TabulatedGrid:
    if roots is None:
        raise ValueError("roots must be provided (call via visitor interface).")

    atom = roots[0]
    line = roots[-1]
    lambda0 = compute_lambda0(atom, line)
    nu0 = lambda0.to(u.Hz, equivalencies=u.spectral())
    n_lambda = quad.n_lambda
    # NOTE(cmo): Should always be odd.
    if n_lambda % 2 == 0:
        n_lambda += 1
    q = np.zeros(n_lambda)

    if quad.q_max <= quad.q0:
        # NOTE(cmo): Linear spacing
        dq = 2.0 * quad.q_max / (n_lambda - 1)
        q = np.arange(n_lambda) * dq - quad.q_max
    else:
        # NOTE(cmo): q_max and q0 are validated to be >= 0.
        a = 10.0 ** (quad.q0 + 0.5)
        x_max = np.log10(a * max(0.5, quad.q_max - quad.q0 - 0.5))
        dx = 2 * x_max / (n_lambda - 1)
        x = np.arange(n_lambda) * dx - x_max
        x10 = 10**x
        q = x + (x10 - 1.0 / x10) / a

    nu = (nu0 * (1.0 + q * quad.q_norm / const.c)).to(u.Hz)
    # NOTE(cmo): We subtract lambda0 due to the form of TabulatedGrid here.
    wavelengths = np.sort(nu.to(u.nm, equivalencies=u.spectral())) - lambda0
    return TabulatedGrid(
        type="Tabulated",
        wavelengths=wavelengths,
    )
