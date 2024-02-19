from typing import Any, Callable, Dict, Mapping, Optional, TypeVar, Union
from typing_extensions import Protocol
import astropy.units as u
from lightweaver.atomic_model import (
    AtomicModel as LwAtomicModel,
    AtomicLevel as LwAtomicLevel,
    AtomicLine as LwAtomicLine,
    LineQuadrature as LwLineQuadrature,
    VoigtLine as LwVoigtLine,
    AtomicContinuum as LwAtomicContinuum,
    HydrogenicContinuum as LwHydrogenicContinuum,
    ExplicitContinuum as LwExplicitContinuum,
    LineType as LwLineType,
    LinearCoreExpWings as LwLinearCoreExpWings,
)
from lightweaver.broadening import (
    LineBroadening as LwLineBroadening,
    LineBroadener as LwLineBroadener,
    RadiativeBroadening as LwRadiativeBroadening,
    VdwUnsold as LwVdwUnsold,
    MultiplicativeStarkBroadening as LwMultiplicativeStarkBroadening,
    QuadraticStarkBroadening as LwQuadraticStarkBroadening,
    HydrogenLinearStarkBroadening as LwHydrogenicStarkBroadening,
)
from lightweaver.collisional_rates import (
    CollisionalRates as LwCollisionalRates,
    Omega as LwOmega,
    CI as LwCI,
    CE as LwCE,
    CP as LwCP,
    CH as LwCH,
    ChargeExchangeNeutralH as LwChargeExchangeNeutralH,
    ChargeExchangeProton as LwChargeExchangeProton,
)
import lightweaver as lw
import numpy as np

from crtaf.core_types import (
    Atom,
    AtomicLevel,
    AtomicBoundBound,
    Element,
    LineBroadening,
    Metadata,
    VoigtBoundBound,
    PrdVoigtBoundBound,
    NaturalBroadening,
    VdWUnsold,
    StarkLinearSutton,
    StarkQuadratic,
    StarkMultiplicative,
    AtomicBoundFree,
    HydrogenicBoundFree,
    TabulatedBoundFree,
    TransCollisionalRates,
    CollisionalRate,
    OmegaRate,
    CIRate,
    CERate,
    CPRate,
    CHRate,
    ChargeExcHRate,
    ChargeExcPRate,
    Fraction as CrtafFraction,
    WavelengthGrid,
)
from crtaf.exts.linear_core_exp_wings_grid import LinearCoreExpWings
from crtaf.spec_version import spec_version


Level_T = TypeVar("Level_T", contravariant=True)


class LevelConversionFn(Protocol[Level_T]):
    """Specification of AtomicLevel conversion function."""

    def __call__(self, l: Level_T) -> AtomicLevel: ...


Broadener_T = TypeVar("Broadener_T", contravariant=True)


class BroadeningConversionFn(Protocol[Broadener_T]):
    """Specification of LineBroadener conversion function."""

    def __call__(self, b: Broadener_T) -> LineBroadening: ...


WavelengthGrid_T = TypeVar("WavelengthGrid_T", contravariant=True)


class WavelengthGridConversionFn(Protocol[WavelengthGrid_T]):
    """Specification of LineQuadrature conversion function."""

    def __call__(self, q: WavelengthGrid_T) -> WavelengthGrid: ...


Line_T = TypeVar("Line_T", contravariant=True)


class LineConversionFn(Protocol[Line_T]):
    """Specification of AtomicLine conversion function.

    Parameters
    ----------
    l : AtomicLine
        The line to convert.
    level_dict: dict of int to str
        Mapping from the level index on the original model to the CRTAF name.
    broadening_types: dict of type to BroadeningConversionFn
        Dispatch dictionary for converting broadening types.
    wavelength_grid_types: dict of type to WavelengthGridConversionFn
        Dispatch dictionary for converting line quadrature types.
    """

    def __call__(
        self,
        l: Line_T,
        level_dict: Dict[int, str],
        broadening_types: Mapping[type, BroadeningConversionFn],
        wavelength_grid_types: Mapping[type, WavelengthGridConversionFn],
    ) -> AtomicBoundBound: ...


Cont_T = TypeVar("Cont_T", contravariant=True)


class ContinuumConversionFn(Protocol[Cont_T]):
    """Specification of AtomicContinuum conversion function.

    Parameters
    ----------
    cont : AtomicContinuum
        The continuum to convert.
    level_dict: dict of int to str
        Mapping from the level index on the original model to the CRTAF name.
    """

    def __call__(
        self,
        cont: Cont_T,
        level_dict: Dict[int, str],
    ) -> AtomicBoundFree: ...


Coll_T = TypeVar("Coll_T", contravariant=True)


class CollisionConversionFn(Protocol[Coll_T]):
    """Specification of CollisionalRate conversion function."""

    def __call__(
        self,
        c: Coll_T,
    ) -> CollisionalRate: ...


def convert_level_from_lw(l: LwAtomicLevel) -> AtomicLevel:
    J = None
    if l.J is not None:
        J = CrtafFraction(numerator=l.J.numerator, denominator=l.J.denominator)
    L = None
    if l.L is not None:
        L = l.L
    S = None
    if l.S is not None:
        S = CrtafFraction(numerator=l.S.numerator, denominator=l.S.denominator)

    return AtomicLevel(
        energy=l.E << (u.cm**-1),
        g=int(l.g),
        stage=l.stage + 1,
        label=l.label,
        J=J,
        L=L,
        S=S,
    )


def convert_voigt_line_from_lw(
    l: LwVoigtLine,
    level_dict: Dict[int, str],
    broadening_types: Mapping[type, BroadeningConversionFn],
    wavelength_grid_types: Mapping[type, WavelengthGridConversionFn],
):
    i = level_dict[l.i]
    j = level_dict[l.j]

    def convert_broadening(b: LwLineBroadener):
        if b.__class__ in broadening_types:
            return broadening_types[b.__class__](b)

        for ty, fn in broadening_types.items():
            if isinstance(b, ty):
                return fn(b)

        raise ValueError(
            f"Could not find any broadening types that match {b.__class__!r}."
        )

    broadening = []
    for b in l.broadening.natural:
        broad = convert_broadening(b)
        broad.elastic = False
        broadening.append(broad)
    for b in l.broadening.elastic:
        broad = convert_broadening(b)
        broad.elastic = True
        broadening.append(broad)
    if l.broadening.other is not None:
        for b in l.broadening.other:
            broad = convert_broadening(b)
            broad.elastic = False
            broadening.append(broad)

    wavelength_grid = None
    if l.quadrature.__class__ in wavelength_grid_types:
        wavelength_grid = wavelength_grid_types[l.quadrature.__class__](l.quadrature)
    else:
        for ty, fn in wavelength_grid_types.items():
            if isinstance(l.quadrature, ty):
                wavelength_grid = fn(l.quadrature)
                break
        else:
            raise ValueError(
                f"Could not find any spectral quadratures types that match {l.quadrature.__class__!r}."
            )

    if l.type == LwLineType.PRD:
        return PrdVoigtBoundBound(
            type="PRD-Voigt",
            transition=[j, i],
            f_value=l.f,
            broadening=broadening,
            wavelength_grid=wavelength_grid,
        )
    return VoigtBoundBound(
        type="Voigt",
        transition=[j, i],
        f_value=l.f,
        broadening=broadening,
        wavelength_grid=wavelength_grid,
    )


def convert_natural_broadening_from_lw(b: LwRadiativeBroadening):
    return NaturalBroadening(type="Natural", elastic=False, value=b.gamma << (u.s**-1))


def convert_stark_linear_broadening_from_lw(b: LwHydrogenicStarkBroadening):
    return StarkLinearSutton(
        type="Stark_Linear_Sutton",
        elastic=True,
    )


def convert_multiplicative_stark_from_lw(b: LwMultiplicativeStarkBroadening):
    return StarkMultiplicative(
        type="Stark_Multiplicative", elastic=True, C_4=b.coeff << (u.Unit("m3 / s"))  # type: ignore
    )


def convert_quadratic_stark_from_lw(b: LwQuadraticStarkBroadening):
    return StarkQuadratic(
        type="Stark_Quadratic",
        elastic=True,
        scaling=b.coeff,
    )


def convert_vdw_unsold_from_lw(b: LwVdwUnsold):
    return VdWUnsold(
        type="VdW_Unsold", elastic=True, H_scaling=b.vals[0], He_scaling=b.vals[1]
    )


def convert_linear_core_exp_wings_from_lw(q: LwLinearCoreExpWings):
    return LinearCoreExpWings(
        type="LinearCoreExpWings",
        q_core=q.qCore,
        q_wing=q.qWing,
        n_lambda=q.Nlambda,
        vmicro_char=3.0 << (u.km / u.s),
    )


def convert_hydrogenic_cont_from_lw(
    cont: LwHydrogenicContinuum, level_dict: Dict[int, str]
):
    i = level_dict[cont.i]
    j = level_dict[cont.j]
    return HydrogenicBoundFree(
        type="Hydrogenic",
        transition=[j, i],
        sigma_peak=cont.alpha0 << (u.m**2),
        lambda_min=cont.minLambda << u.nm,
        n_lambda=cont.NlambdaGen,
    )


def convert_tabulated_cont_from_lw(
    cont: LwExplicitContinuum, level_dict: Dict[int, str]
):
    i = level_dict[cont.i]
    j = level_dict[cont.j]
    return TabulatedBoundFree(
        type="Tabulated",
        transition=[j, i],
        wavelengths=np.array(cont.wavelengthGrid) << u.nm,
        sigma=np.array(cont.alphaGrid) << (u.m**2),
    )


def convert_omega_from_lw(r: LwOmega):
    return OmegaRate(
        type="Omega",
        temperature=np.array(r.temperature) << u.K,
        data=np.array(r.rates) << u.Unit(""),
    )


def convert_basic_electron_rate_from_lw(r: Union[LwCE, LwCI]):
    if isinstance(r, LwCE):
        rate = CERate
        typename = "CE"
    elif isinstance(r, LwCI):
        rate = CIRate
        typename = "CI"
    else:
        raise TypeError(f"Expected CI or CERate, got {type(r)!r}")

    return rate(
        type=typename,  # type: ignore
        temperature=np.array(r.temperature) << u.K,
        data=np.array(r.rates) << u.Unit("m3 s-1 K(-1/2)"),
    )


def convert_basic_collision_rate_from_lw(
    r: Union[LwCH, LwCP, LwChargeExchangeNeutralH, LwChargeExchangeProton]
):
    if isinstance(r, LwCH):
        rate = CHRate
        typename = "CH"
    elif isinstance(r, LwCP):
        rate = CPRate
        typename = "CP"
    elif isinstance(r, LwChargeExchangeNeutralH):
        rate = ChargeExcHRate
        typename = "ChargeExcH"
    elif isinstance(r, LwChargeExchangeProton):
        rate = ChargeExcPRate
        typename = "ChargeExcP"
    else:
        raise TypeError(
            f"Expected CH, CP, ChargeExchangeNeutralH, or ChargeExchangeProtonRate, got {type(r)!r}"
        )
    return rate(
        type=typename,  # type: ignore
        temperature=np.array(r.temperature) << u.K,
        data=np.array(r.rates) << u.Unit("m3 s-1"),
    )


def default_level_types() -> Dict[type, LevelConversionFn]:
    return {LwAtomicLevel: convert_level_from_lw}


def default_line_types() -> Dict[type, LineConversionFn]:
    return {LwVoigtLine: convert_voigt_line_from_lw}


def default_broadening_types() -> Dict[type, BroadeningConversionFn]:
    return {
        LwRadiativeBroadening: convert_natural_broadening_from_lw,
        LwHydrogenicStarkBroadening: convert_stark_linear_broadening_from_lw,
        LwMultiplicativeStarkBroadening: convert_multiplicative_stark_from_lw,
        LwQuadraticStarkBroadening: convert_quadratic_stark_from_lw,
        LwVdwUnsold: convert_vdw_unsold_from_lw,
    }


def default_wavelength_grid_types():
    return {LwLinearCoreExpWings: convert_linear_core_exp_wings_from_lw}


def default_continuum_types():
    return {
        LwHydrogenicContinuum: convert_hydrogenic_cont_from_lw,
        LwExplicitContinuum: convert_tabulated_cont_from_lw,
    }


def default_collision_types():
    return {
        LwOmega: convert_omega_from_lw,
        LwCI: convert_basic_electron_rate_from_lw,
        LwCE: convert_basic_electron_rate_from_lw,
        LwCH: convert_basic_collision_rate_from_lw,
        LwCP: convert_basic_collision_rate_from_lw,
        LwChargeExchangeNeutralH: convert_basic_collision_rate_from_lw,
        LwChargeExchangeProton: convert_basic_collision_rate_from_lw,
    }


# https://stackoverflow.com/a/40274588
def to_roman_numeral(num: int) -> str:
    num_map = [
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ]

    result = []
    while num > 0:
        for value, roman_char in num_map:
            while num >= value:
                result.append(roman_char)
                num -= value

    return "".join(result)


class LightweaverAtomConverter:
    """Convert a Lightweaver model atom to CRTAF format.
    By default uses the core set of CRTAF functions (and therefore cannot
    convert every Lightweaver atom), but can be overidden with the optional
    dicts passed to this class.

    Parameters
    ----------
    level_types : dict mapping type to LevelConversionFn
        Mapping from a Lightweaver type representing an atomic level to the
        equivalent CRTAF type.
    line_types : dict mapping type to LineConversionFn
        Mapping from a Lightweaver type representing a bound-bound transition to
        the equivalent CRTAF type.
    broadening_types : dict mapping type to BroadeningConversionFn
        Mapping from a Lightweaver type representing line broadening to the
        equivalent CRTAF type.
    wavelength_grid_types : dict mapping type to WavelengthGridConversionFn
        Mapping from a Lightweaver type representing line spectral quadrature to the
        equivalent CRTAF type.
    continuum_types : dict mapping type to ContinuumConversionFn
        Mapping from a Lightweaver type representing a bound-free transition to
        the equivalent CRTAF type.
    collision_types : dict mapping type to CollisionConversionFn
        Mapping from a Lightweaver type representing a collisional-rate to
        the equivalent CRTAF type.
    """

    def __init__(
        self,
        level_types: Optional[Mapping[type, LevelConversionFn]] = None,
        line_types: Optional[Mapping[type, LineConversionFn]] = None,
        broadening_types: Optional[Mapping[type, BroadeningConversionFn]] = None,
        wavelength_grid_types: Optional[
            Mapping[type, WavelengthGridConversionFn]
        ] = None,
        continuum_types: Optional[Mapping[type, ContinuumConversionFn]] = None,
        collision_types: Optional[Mapping[type, CollisionConversionFn]] = None,
    ):
        if level_types is None:
            level_types = default_level_types()
        self.level_types = level_types

        if line_types is None:
            line_types = default_line_types()
        self.line_types = line_types

        if broadening_types is None:
            broadening_types = default_broadening_types()
        self.broadening_types = broadening_types

        if wavelength_grid_types is None:
            wavelength_grid_types = default_wavelength_grid_types()
        self.wavelength_grid_types = wavelength_grid_types

        if continuum_types is None:
            continuum_types = default_continuum_types()
        self.continuum_types = continuum_types

        if collision_types is None:
            collision_types = default_collision_types()
        self.collision_types = collision_types

    def convert(self, atom: LwAtomicModel) -> Atom:
        """Convert the provided Lightweaver model to CRTAF format, using the
        methods attached to this object.
        """
        levels = {}
        level_conversion_dict = {}
        current_stage = 1
        stage_change_idx = 0
        for idx, l in enumerate(atom.levels):
            crtaf_level = self.dispatch(self.level_types, l)

            if l.stage + 1 != current_stage:
                current_stage = l.stage + 1
                stage_change_idx = idx

            label = f"{to_roman_numeral(current_stage)}_{idx-stage_change_idx+1}"
            levels[label] = crtaf_level
            level_conversion_dict[idx] = label

        lines = []
        for line in atom.lines:
            crtaf_line = self.dispatch(
                self.line_types,
                line,
                level_dict=level_conversion_dict,
                broadening_types=self.broadening_types,
                wavelength_grid_types=self.wavelength_grid_types,
            )
            lines.append(crtaf_line)

        continua = []
        for cont in atom.continua:
            crtaf_cont = self.dispatch(
                self.continuum_types,
                cont,
                level_dict=level_conversion_dict,
            )
            continua.append(crtaf_cont)

        collisions_seen = []
        collisions = []
        for coll in atom.collisions:
            key = (coll.j, coll.i)
            if key in collisions_seen:
                continue

            collisions_seen.append(key)

            collisions_with_key = [
                c for c in atom.collisions if c.j == coll.j and c.i == coll.i
            ]
            crtaf_colls = [
                self.dispatch(self.collision_types, c) for c in collisions_with_key
            ]

            collisions.append(
                TransCollisionalRates(
                    transition=[
                        level_conversion_dict[key[0]],
                        level_conversion_dict[key[1]],
                    ],
                    data=crtaf_colls,
                )
            )

        abundance = (
            np.log10(
                lw.DefaultAtomicAbundance[atom.element] / lw.DefaultAtomicAbundance["H"]
            )
            + 12.0
        )
        element = Element(
            symbol=atom.element.name,
            atomic_mass=atom.element.mass,
            abundance=abundance,
            Z=atom.element.Z,
        )

        meta = Metadata(
            version=spec_version,
            level="high-level",
            extensions=["linear_core_exp_wings"],
        )

        return Atom(
            crtaf_meta=meta,
            element=element,
            levels=levels,
            lines=lines,
            continua=continua,
            collisions=collisions,
        )

    @staticmethod
    def dispatch(methods: Mapping[type, Any], obj: Any, *args, **kwargs):
        """Dispatch to the best matching method in methods dict or raise an Error.

        Parameters
        ----------
        methods : Dict of type to callable
            The methods to dispatch over.
        obj : Any
            The object to use for type information.
        *args, **kwargs : Any
            Additional arguments passed to the conversion.
        """
        if obj.__class__ in methods:
            return methods[obj.__class__](obj, *args, **kwargs)

        for ty, fn in methods.items():
            if isinstance(obj, ty):
                return fn(obj, *args, **kwargs)

        raise ValueError(
            f"Could not find any conversion types that match {obj.__class__!r}."
        )
