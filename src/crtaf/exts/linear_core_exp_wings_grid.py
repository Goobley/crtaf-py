from typing import Any, Final, List, Literal, Optional, ClassVar
import numpy as np
from pydantic import field_validator
from crtaf import WavelengthGrid, AstropyQty, TabulatedGrid
import astropy.constants as const

from crtaf.physics_utils import compute_lambda0


class LinearCoreExpWings(WavelengthGrid, type_name="LinearCoreExpWings"):
    """
    RH-Style line quadrature, with approximately linear core spacing and
    exponential wing spacing, by using a function of the form
    q(n) = a*(n + (exp(b*n)-1))
    with n in [0, N) satisfying the following conditions:

     - q[0] = 0

     - q[(N-1)/2] = q_core

     - q[N-1] = q_wing.

    If q_wing <= 2 * q_core, linear grid spacing will be used for this transition.
    """

    _crtaf_ext_name: ClassVar[str] = "linear_core_exp_wings"

    type: str = Literal["LinearCoreExpWings"]  # type: ignore
    q_core: float
    q_wing: float
    n_lambda: int
    vmicro_char: AstropyQty

    @field_validator("vmicro_char")
    @classmethod
    def _validate(cls, v):
        v.to("m / s")
        return v


def simplify_linear_core_exp_wings(
    quad: LinearCoreExpWings, roots: Optional[List[Any]], visitor, *args, **kwargs
) -> TabulatedGrid:
    if roots is None:
        raise ValueError("roots must be provided (call via visitor interface).")
    atom = roots[0]
    line = roots[-1]

    beta = 1.0
    if quad.q_wing > 2.0 * quad.q_core:
        beta = quad.q_wing / (2.0 * quad.q_core)
    n_lambda_half = quad.n_lambda // 2
    n_lambda_half += 1

    y = beta + np.sqrt(beta**2 + (beta - 1.0) * n_lambda_half + 2.0 - 3.0 * beta)
    b = 2.0 * np.log(y) / (n_lambda_half - 1.0)
    a = quad.q_wing / (n_lambda_half - 2.0 + y**2)
    n_grid = np.arange(n_lambda_half)
    q = a * (n_grid + (np.exp(b * n_grid) - 1.0))

    n_lambda_full = 2 * n_lambda_half - 1
    n_mid = n_lambda_half - 1
    q_grid = np.zeros(n_lambda_full)

    q_grid[:n_mid][::-1] = -q[1:]
    q_grid[n_mid + 1 :] = q[1:]

    lambda0 = compute_lambda0(atom, line)
    q_to_lambda = lambda0 * quad.vmicro_char / const.c
    result = q_grid * q_to_lambda
    # NOTE(cmo): TabulatedGrid expects grid relative to lambda0, so don't add it
    # here like we normally would.
    return TabulatedGrid(type="Tabulated", wavelengths=result.to("nm"))
