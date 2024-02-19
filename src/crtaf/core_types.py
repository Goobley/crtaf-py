from copy import copy
from io import StringIO
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Union
from typing_extensions import Annotated
from annotated_types import Gt, Le
import numpy as np
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    SerializeAsAny,
    StringConstraints,
    ValidatorFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.config import ConfigDict
import pydantic_core.core_schema as core_schema
from pydantic_numpy import NpNDArrayFp64
import astropy.units as u
import ruamel.yaml
from ruamel.yaml import CommentedMap, CommentedSeq
from ruamel.yaml.comments import CommentedBase

from crtaf.spec_version import spec_version


class IterateQuantitiesMixin:
    """Simple mixin to allow for keeping track of all Quantities on a type and
    mapping unit conversions. Needs to be applied to a pydantic model`"""

    def qty_children(self):
        for k, v in self:  # type: ignore
            if isinstance(v, IterateQuantitiesMixin):
                yield v

    def named_qty_children(self):
        for k, v in self:  # type: ignore
            if isinstance(v, IterateQuantitiesMixin):
                yield k, v

    def quantities(self):
        for k, v in self:  # type: ignore
            if isinstance(v, u.Quantity):
                yield v

    def named_quantities(self):
        for k, v in self:  # type: ignore
            if isinstance(v, u.Quantity):
                yield k, v

    def apply_unit_conversion(
        self, operation: Callable[[u.Quantity], Optional[u.Quantity]]
    ):
        for k, v in self.named_quantities():
            result = operation(v)
            if result is not None:
                self.__setattr__(k, result)

        for child in self.qty_children():
            child.apply_unit_conversion(operation)


class SimplifyAtomicStructureMixin:
    def simplify(self):
        raise NotImplementedError()


class AtomicSimplificationVisitor:
    """
    Handler to simplify a model atom using the provided set of visitors.

    Parameters
    ----------
    visitors : dict
        Mapping of type to function that simplifies that type (potentially just
        ensuring units are correct).
    extensions : list of str
        The names of extensions to accept (not simplify) on the model.
    """

    def __init__(
        self, visitors: Dict[type, Callable], extensions: Optional[List[str]] = None
    ):
        self.visitors = visitors
        if extensions is None:
            extensions = []
        self.extensions = extensions
        self.extensions_encountered = set()

    def visit(
        self, obj: "CrtafBaseModel", *args, accept_failure: bool = False, **kwargs
    ) -> Any:
        """
        Apply the visitors to the model atom. Extra arguments are passed onto
        the visitors.

        Parameters
        ----------
        obj : CrtafBaseModel
            The component to simplify (call with the Atom by default).
        accept_failure : bool, optional
            Whether to maintain objects that don't have a visitor provided and
            aren't in the extension list. Default, False.
        """
        visit_fn = None
        if type(obj) in self.visitors:
            visit_fn = self.visitors[type(obj)]
        else:
            for cls in obj.__class__.__mro__:
                if cls in self.visitors:
                    visit_fn = self.visitors[cls]
                    break

        ext_name = None
        if hasattr(obj, "_crtaf_ext_name"):
            ext_name = obj._crtaf_ext_name  # type: ignore

        if ext_name is not None and ext_name in self.extensions:
            result = obj
        elif visit_fn is None:
            if accept_failure:
                result = obj
            else:
                raise ValueError(f"No valid visitor for type {type(obj)!r}")
        else:
            result = visit_fn(
                obj, *args, visitor=self, accept_failure=accept_failure, **kwargs
            )

        if hasattr(result, "_crtaf_ext_name"):
            self.extensions_encountered.add(result._crtaf_ext_name)  # type: ignore
        return result


class CrtafBaseModel(BaseModel):
    """
    Class used to override default behaviour on model_dump, and add yaml functions to
    allow for formatting control and simplify converting to string.
    """

    def model_dump(self, *args, exclude_none: bool = True, **kwargs):
        return super().model_dump(*args, exclude_none=exclude_none, **kwargs)

    @staticmethod
    def _convert_element(v: Any):
        if isinstance(v, dict):
            return CrtafBaseModel._recursive_dict_convert(v)

        if isinstance(v, list):
            return CrtafBaseModel._recursive_list_convert(v)

        return v

    @staticmethod
    def _recursive_dict_convert(d: Dict[str, Any]) -> CommentedMap:
        result = CommentedMap()
        for k, v in d.items():
            result[k] = CrtafBaseModel._convert_element(v)
        return result

    @staticmethod
    def _recursive_list_convert(l: List[Any]) -> CommentedSeq:
        result = CommentedSeq()
        for v in l:
            result.append(CrtafBaseModel._convert_element(v))
        return result

    @staticmethod
    def _recursive_list_format(
        attr: List[Any], serialised: CommentedSeq
    ) -> CommentedSeq:
        result = CommentedSeq()
        for i, f in enumerate(attr):
            if isinstance(f, CrtafBaseModel):
                if not isinstance(serialised[i], (dict, CommentedMap)):
                    raise ValueError(
                        f"model_dump failed to serialise to dict, got {type(serialised[i])!r}."
                    )
                result.append(CrtafBaseModel._recursive_yaml_format(f, serialised[i]))
            elif isinstance(f, list):
                result.append(CrtafBaseModel._recursive_list_format(f, serialised[i]))
            else:
                result.append(serialised[i])
        return result

    def _recursive_yaml_format(self, d: Union[dict, CommentedMap]) -> CommentedMap:
        """
        Recurses through the YAML tree and formatting the CommentedMap and
        CommentedSeq as prescribed by `format_yaml_dict`.
        """
        if isinstance(d, CommentedBase):
            d = copy(d)
        else:
            d = self._recursive_dict_convert(copy(d))

        for k, v in d.items():
            if hasattr(self, k):
                attr = getattr(self, k)
                if isinstance(attr, CrtafBaseModel):
                    if not isinstance(v, dict):
                        raise ValueError(
                            f"The result of model_dump from a BaseModel should be a dict, got {type(attr)!r}"
                        )
                    dd = attr._recursive_yaml_format(d[k])
                    d[k] = dd
                elif isinstance(attr, list):
                    d[k] = self._recursive_list_format(attr, d[k])
        return self.format_yaml_dict(d)

    def yaml_dict(self, *args, **kwargs) -> CommentedMap:
        """CommentedMap (dict with formatting), set up for pretty-printing via
        ruamel.yaml. Effectively the same as calling `model_dump` but with this
        extra formatting."""
        model = self.model_dump(*args, **kwargs)
        result = self._recursive_yaml_format(model)
        return result

    def format_yaml_dict(self, c: CommentedMap) -> CommentedMap:
        """Modify the formatting of a yaml_dict to get a tidy representation.
        Modifies c in place but also returns it. Base implementation leaves as is."""

        return c

    def yaml_dumps(self, *args, **kwargs) -> str:
        """YAML string representation of the model."""
        stream = StringIO()
        yaml = ruamel.yaml.YAML(typ="rt")
        data = self.yaml_dict()
        yaml.dump(data, stream)
        return stream.getvalue()

    @classmethod
    def yaml_loads(cls, inp: str):
        """Construct from YAML string representation."""
        yaml = ruamel.yaml.YAML(typ="rt")
        data = yaml.load(inp)
        return cls.model_validate(data)


class PolymorphicBaseModel(CrtafBaseModel):
    _is_polymorphic_base: ClassVar = True
    _registry: ClassVar = {}

    def __init_subclass__(
        cls,
        type_name: str = "",
        is_polymorphic_base: bool = False,
        **kwargs,
    ):
        if is_polymorphic_base:
            # NOTE(cmo): Set up a registry for this instance -- since we're forming multiple polymorphic "trees" of classes for different objects.
            cls._registry = {}
        else:
            if type_name == "":
                raise ValueError("Derived polymorphic models must provide a type name.")
            cls._registry[type_name] = cls
        cls._is_polymorphic_base = is_polymorphic_base
        return super().__init_subclass__(**kwargs)

    # NOTE(cmo): Inspired by
    # https://github.com/pydantic/pydantic/discussions/7008#discussioncomment-6826052
    @model_validator(mode="wrap")
    @classmethod
    def _reify_type(cls, v: Any, handler: ValidatorFunctionWrapHandler):
        # NOTE(cmo): If it's already an object pass it to the default validator
        if not isinstance(v, dict):
            return handler(v)

        # NOTE(cmo): If someone has requested a derived type directly, just pass it over.
        if not cls._is_polymorphic_base:
            return handler(v)

        # NOTE(cmo): Otherwise lookup the type in the dict and registry.
        t = v["type"]
        if t not in cls._registry:
            # TODO(cmo): Do a levenstein lookup for suggestion?
            raise ValueError(
                f"Type {t} is not in the known types for {cls!r} ({cls._registry.keys()})"
            )

        return cls._registry[t].model_validate(v)


class DimensionalQuantity(CrtafBaseModel):
    unit: str
    value: Union[float, NpNDArrayFp64]

    @field_serializer("value")
    def serialize_value(self, value, _info):
        return value.tolist()


class _AstropyQtyAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        _source_type: Any,
        _handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        def validate_qty(qty: DimensionalQuantity) -> u.Quantity:
            validated_unit = u.Unit(qty.unit)
            return qty.value << validated_unit  # type: ignore

        def serialise_qty(qty: u.Quantity):
            value = qty.value
            if len(value.shape) > 0:
                value = value.tolist()
            else:
                value = float(value)
            unit = qty.unit
            return {
                "unit": unit.to_string(),  # type: ignore
                "value": value,
            }

        from_dim_qty_schema = core_schema.chain_schema(
            [
                DimensionalQuantity.__pydantic_core_schema__,
                core_schema.no_info_plain_validator_function(validate_qty),
            ]
        )
        direct_qty_schema = core_schema.is_instance_schema(cls=u.Quantity)

        schema = core_schema.union_schema(
            [
                direct_qty_schema,
                from_dim_qty_schema,
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialise_qty
            ),
        )

        return schema


AstropyQty = Annotated[u.Quantity, _AstropyQtyAnnotation]


class Metadata(CrtafBaseModel):
    version: Annotated[
        str, StringConstraints(strip_whitespace=True, pattern=r"^v\d+\.\d+(\.\d+)*")
    ]
    level: Union[Literal["high-level"], Literal["simplified"]]
    extensions: List[str]
    notes: str = ""


class Element(CrtafBaseModel):
    symbol: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    atomic_mass: Optional[Annotated[float, Gt(0)]] = None
    abundance: Optional[Annotated[float, Le(12.0)]] = None
    Z: Optional[int] = None
    N: Optional[int] = None


class Fraction(CrtafBaseModel):
    numerator: int
    denominator: int


class AtomicLevel(CrtafBaseModel, IterateQuantitiesMixin, SimplifyAtomicStructureMixin):
    energy: AstropyQty
    g: int
    stage: int
    label: Optional[str] = None
    J: Optional[Union[Fraction, float]] = None
    L: Optional[int] = None
    S: Optional[Union[Fraction, float]] = None

    @field_validator("energy")
    @classmethod
    def _validate_energy(cls, v: AstropyQty):
        if v.unit.physical_type == "energy":  # type: ignore
            return v

        # NOTE(cmo): This will raise a unit conversion error if it can't be
        # converted.
        v.to(u.eV, equivalencies=u.spectral())
        return v

    @model_validator(mode="after")
    def _validate_JLS(self):
        conds = [self.J is None, self.L is None, self.S is None]
        if any(conds) != all(conds):
            missing_keys = [s for i, s in enumerate(["J", "L", "S"]) if conds[i]]
            raise ValueError(
                f"If one of J, L, S are defined, all must be. Missing ({missing_keys})."
            )
        return self

    def simplify(self) -> "SimplifiedAtomicLevel":
        return SimplifiedAtomicLevel(
            energy=self.energy,
            energy_eV=self.energy,
            g=self.g,
            stage=self.stage,
            label=self.label,
            J=self.J,
            L=self.L,
            S=self.S,
        )


class SimplifiedAtomicLevel(AtomicLevel):
    energy: AstropyQty
    energy_eV: AstropyQty
    g: int
    stage: int
    label: Optional[str] = None
    J: Optional[Union[Fraction, float]] = None
    L: Optional[int] = None
    S: Optional[Union[Fraction, float]] = None

    @model_validator(mode="after")
    def _validate(self):
        self.energy = self.energy.to(u.cm**-1, equivalencies=u.spectral())
        self.energy_eV = self.energy_eV.to(u.eV, equivalencies=u.spectral())
        return self

    def format_yaml_dict(self, c: CommentedMap) -> CommentedMap:
        c.fa.set_flow_style()
        return c


class LineBroadening(
    PolymorphicBaseModel,
    IterateQuantitiesMixin,
    SimplifyAtomicStructureMixin,
    is_polymorphic_base=True,
):
    type: str
    elastic: Optional[bool] = True

    def simplify(self) -> "LineBroadening":
        return super().simplify()

    def format_yaml_dict(self, c: CommentedMap) -> CommentedMap:
        c.fa.set_flow_style()
        return c


class NaturalBroadening(LineBroadening, type_name="Natural"):
    type: Literal["Natural"]
    elastic: Optional[bool] = False
    value: AstropyQty

    @field_validator("value")
    @classmethod
    def _validate(cls, value: u.Quantity):
        value.to(u.s**-1)
        return value

    def simplify(self):
        new_value = self.value.to(u.s**-1)
        return NaturalBroadening(type=self.type, value=new_value)


class ElasticBroadening(LineBroadening, type_name="__elastic"):
    elastic: Optional[bool] = True


class StarkLinearSutton(ElasticBroadening, type_name="Stark_Linear_Sutton"):
    type: Literal["Stark_Linear_Sutton"]
    n_upper: Optional[Annotated[int, Gt(0)]] = None
    n_lower: Optional[Annotated[int, Gt(0)]] = None

    @model_validator(mode="after")
    def _validate(self):
        conds = [self.n_upper is None, self.n_lower is None]
        if any(conds) != all(conds):
            raise ValueError(
                "Must provide both n_upper and n_lower if providing one or the other"
            )

        if self.n_upper is not None:
            assert self.n_upper > self.n_lower  # type: ignore
            assert self.n_lower > 0  # type: ignore

        return self

    def simplify(self):
        raise NotImplementedError()


class StarkQuadratic(ElasticBroadening, type_name="Stark_Quadratic"):
    type: Literal["Stark_Quadratic"]
    scaling: Optional[float] = 1.0

    def simplify(self):
        raise NotImplementedError()


class StarkMultiplicative(ElasticBroadening, type_name="Stark_Multiplicative"):
    type: Literal["Stark_Multiplicative"]
    C_4: AstropyQty
    scaling: Optional[float] = 1.0

    @field_validator("C_4")
    @classmethod
    def _validate_C_4(cls, value: u.Quantity):
        if value.unit.physical_type != "volumetric flow rate":  # type: ignore
            raise ValueError(
                f"C_4 is expected to be convertible to volumetric flow rate. Got units of {value.unit}"
            )
        return value


class VdWUnsold(ElasticBroadening, type_name="VdW_Unsold"):
    """
    Van der Waals collisional broadening following Unsold.
    See:
        - Traving 1960, "Uber die Theorie der Druckverbreiterung
            von Spektrallinien", p 91-97
        - Mihalas 1978, p. 282ff, and Table 9-1
    """

    type: Literal["VdW_Unsold"]
    H_scaling: float = 1.0
    He_scaling: float = 1.0

    def simplify(self):
        raise NotImplementedError()


class ScaledExponents(LineBroadening, type_name="Scaled_Exponents"):
    type: Literal["Scaled_Exponents"]
    elastic: bool = True
    scaling: float
    temperature_exponent: float
    hydrogen_exponent: float
    electron_exponent: float

    def simplify(self):
        return self


class WavelengthGrid(
    PolymorphicBaseModel,
    IterateQuantitiesMixin,
    SimplifyAtomicStructureMixin,
    is_polymorphic_base=True,
):
    type: str

    def simplify(self) -> "WavelengthGrid":
        raise NotImplementedError()


class LinearGrid(WavelengthGrid, type_name="Linear"):
    type: Literal["Linear"]
    n_lambda: Annotated[int, Gt(0)]
    delta_lambda: AstropyQty

    @field_validator("delta_lambda")
    @classmethod
    def _validate_delta_lambda(cls, value: u.Quantity):
        value.to(u.nm)
        return value

    def simplify(self):
        delta_lambda = self.delta_lambda.to(u.nm)
        return LinearGrid(
            type=self.type, n_lambda=self.n_lambda, delta_lambda=delta_lambda
        )


class TabulatedGrid(WavelengthGrid, type_name="Tabulated"):
    type: Literal["Tabulated"]
    wavelengths: AstropyQty

    @field_validator("wavelengths")
    @classmethod
    def _validate_wavelengths(cls, value: u.Quantity):
        value.to(u.nm)
        if len(value.shape) == 0 or len(value.shape) > 1:
            raise ValueError(
                "Wavelengths should be a 1D array of at least one point (relative to line centre)."
            )

        return value

    def simplify(self):
        wavelengths = self.wavelengths.to(u.nm)
        return TabulatedGrid(type=self.type, wavelengths=wavelengths)

    def format_yaml_dict(self, c: CommentedMap) -> CommentedMap:
        c["wavelengths"].fa.set_flow_style()
        return c


class AtomicBoundBoundImpl(
    PolymorphicBaseModel,
    IterateQuantitiesMixin,
    is_polymorphic_base=True,
):
    type: str
    transition: List[str]

    @field_validator("transition")
    @classmethod
    def _validate_transition(cls, v: Any):
        length = len(v)
        if length != 2:
            raise ValueError(f"Transitions can only be between two levels, got {v}")
        return v


AtomicBoundBound = SerializeAsAny[AtomicBoundBoundImpl]


class VoigtBoundBound(AtomicBoundBound, type_name="Voigt"):
    type: Literal["Voigt"]
    f_value: float
    broadening: List[SerializeAsAny[LineBroadening]]
    wavelength_grid: SerializeAsAny[WavelengthGrid]
    Aji: Optional[AstropyQty] = None
    Bji: Optional[AstropyQty] = None
    Bji_wavelength: Optional[AstropyQty] = None
    Bij: Optional[AstropyQty] = None
    Bij_wavelength: Optional[AstropyQty] = None
    lambda0: Optional[AstropyQty] = None

    def format_yaml_dict(self, c: CommentedMap) -> CommentedMap:
        c["transition"].fa.set_flow_style()
        return c


class PrdVoigtBoundBound(VoigtBoundBound, type_name="PRD-Voigt"):
    type: Literal["PRD-Voigt"]


class AtomicBoundFreeImpl(
    PolymorphicBaseModel,
    IterateQuantitiesMixin,
    SimplifyAtomicStructureMixin,
    is_polymorphic_base=True,
):
    type: str
    transition: List[str]

    @field_validator("transition")
    @classmethod
    def _validate_transition(cls, v: Any):
        length = len(v)
        if length != 2:
            raise ValueError(f"Transitions can only be between two levels, got {v}")
        return v

    def simplify(self) -> "AtomicBoundFreeImpl":
        return super().simplify()


AtomicBoundFree = SerializeAsAny[AtomicBoundFreeImpl]


class HydrogenicBoundFree(AtomicBoundFree, type_name="Hydrogenic"):
    type: Literal["Hydrogenic"]
    sigma_peak: AstropyQty
    lambda_min: AstropyQty
    n_lambda: int

    def simplify(self):
        sigma_peak = self.sigma_peak.to(u.m**2)
        lambda_min = self.lambda_min.to(u.nm)
        return HydrogenicBoundFree(
            type=self.type,
            transition=self.transition,
            sigma_peak=sigma_peak,
            lambda_min=lambda_min,
            n_lambda=self.n_lambda,
        )

    def format_yaml_dict(self, c: CommentedMap) -> CommentedMap:
        c["transition"].fa.set_flow_style()
        return c


class TabulatedBoundFreeIntermediate(BaseModel):
    """Intermediate type used in parsing. Not of any use in a model."""

    type: Literal["Tabulated"]
    transition: List[str]
    unit: List[str]
    value: NpNDArrayFp64

    @field_validator("unit")
    @classmethod
    def _validate_unit_list(cls, v: Any):
        length = len(v)
        if length != 2:
            raise ValueError(f"Expected 2 units, got {length}.")
        return v

    @field_validator("value")
    @classmethod
    def _validate_value(cls, v: Any):
        if len(v.shape) != 2:
            raise ValueError(f"Expected 2D array, got {len(v.shape)}D.")
        if v.shape[0] < 2:
            raise ValueError(
                f"Expected at least 2 rows of (wavelength, cross-section), got {v.shape[0]}."
            )
        return v


class TabulatedBoundFree(AtomicBoundFree, type_name="Tabulated"):
    type: Literal["Tabulated"]
    wavelengths: AstropyQty
    sigma: AstropyQty

    @model_validator(mode="wrap")
    @classmethod
    def _validate(cls, v: Any, handler: ValidatorFunctionWrapHandler):
        if not isinstance(v, dict):
            return handler(v)

        # NOTE(cmo): This validator can end up being called twice, on an already
        # adjusted dict, catch that here and pass straight to handler.
        if "unit" not in v and "value" not in v:
            return handler(v)

        intermediate = TabulatedBoundFreeIntermediate.model_validate(v)
        wavelength_unit = u.Unit(intermediate.unit[0])
        assert wavelength_unit.is_equivalent(  # type: ignore
            u.m, equivalencies=u.spectral()
        ), "Wavelength unit (first unit) is not convertible to length."
        sigma_unit = u.Unit(intermediate.unit[1])
        assert sigma_unit.is_equivalent(  # type: ignore
            "m2"
        ), "Cross-section unit (second unit) is not convertible to area."
        wavelengths = np.ascontiguousarray(intermediate.value[:, 0])  # type: ignore
        sigma = np.ascontiguousarray(intermediate.value[:, 1])  # type: ignore

        return handler(
            {
                "type": intermediate.type,
                "transition": intermediate.transition,
                "wavelengths": wavelengths << wavelength_unit,  # type: ignore
                "sigma": sigma << sigma_unit,  # type: ignore
            }
        )

    @model_serializer
    def _serialise(self):
        value = np.hstack((self.wavelengths.value[:, None], self.sigma.value[:, None]))
        value = value.tolist()
        unit = [self.wavelengths.unit.to_string(), self.sigma.unit.to_string()]  # type: ignore
        return {
            "type": self.type,
            "transition": self.transition,
            "unit": unit,
            "value": value,
        }

    def simplify(self):
        wavelengths = self.wavelengths.to(u.nm)
        sigma = self.sigma.to(u.m**2)
        return TabulatedBoundFree(
            type=self.type,
            transition=self.transition,
            wavelengths=wavelengths,
            sigma=sigma,
        )

    def format_yaml_dict(self, c: CommentedMap) -> CommentedMap:
        c["transition"].fa.set_flow_style()
        for l in c["value"]:
            l.fa.set_flow_style()
        return c


class CollisionalRateImpl(
    PolymorphicBaseModel,
    IterateQuantitiesMixin,
    SimplifyAtomicStructureMixin,
    is_polymorphic_base=True,
):
    type: str


CollisionalRate = SerializeAsAny[CollisionalRateImpl]


class TemperatureInterpolationRateImpl(
    CollisionalRateImpl,
    IterateQuantitiesMixin,
    SimplifyAtomicStructureMixin,
    type_name="__temperature_interpolation",
):
    type: str
    temperature: AstropyQty
    data: AstropyQty

    @model_validator(mode="after")
    def _validate_lengths(self):
        if self.temperature.shape != self.data.shape:
            raise ValueError(
                "Temperature and Data must have the same shape for a TemperatureInterpolationRate."
            )
        if self.temperature.ndim != 1:
            raise ValueError(
                "Temperature and Data must be one-dimensional for a TemperatureInterpolationRate."
            )
        if self.temperature.shape[0] < 2:
            raise ValueError(
                "Temperature and Data must have at least two entries for a TemperatureInterpolationRate."
            )

        return self

    def simplify(self) -> "TemperatureInterpolationRateImpl":
        return super().simplify()

    def format_yaml_dict(self, c: CommentedMap) -> CommentedMap:
        c["temperature"]["value"].fa.set_flow_style()
        c["data"]["value"].fa.set_flow_style()
        return c


TemperatureInterpolationRate = SerializeAsAny[TemperatureInterpolationRateImpl]


class OmegaRate(TemperatureInterpolationRate, type_name="Omega"):
    """
    Collisional excitation of ions by electrons. Seaton's dimensionless
    collision strength Omega as a function of temperature.

    Scales as 1/(sqrt T) exp(DeltaE)
    """

    type: Literal["Omega"]

    def simplify(self) -> "OmegaRate":
        temperature = self.temperature.to(u.K)
        return OmegaRate(type="Omega", temperature=temperature, data=self.data)


class CIRate(TemperatureInterpolationRate, type_name="CI"):
    """
    Collisional ionisation of ions by electrons.
    Units of s-1 K(-1/2) m3

    Scales as 1/(sqrt T) exp(DeltaE)
    """

    type: Literal["CI"]

    @field_validator("data")
    @classmethod
    def _validate(cls, v: AstropyQty):
        v.to("s-1 K(-1/2) m3")
        return v

    def simplify(self) -> "CIRate":
        temperature = self.temperature.to(u.K)
        data = self.data.to("s-1 K(-1/2) m3")
        return CIRate(type="CI", temperature=temperature, data=data)


class CERate(TemperatureInterpolationRate, type_name="CE"):
    """
    Collisional excitation of neutrals by electrons.
    Units of s-1 K(-1/2) m3

    Scales as 1/(sqrt T) exp(DeltaE)
    """

    type: Literal["CE"]

    @field_validator("data")
    @classmethod
    def _validate(cls, v: AstropyQty):
        v.to("s-1 K(-1/2) m3")
        return v

    def simplify(self) -> "CERate":
        temperature = self.temperature.to(u.K)
        data = self.data.to("s-1 K(-1/2) m3")
        return CERate(type="CE", temperature=temperature, data=data)


class CPRate(TemperatureInterpolationRate, type_name="CP"):
    """
    Collisional excitation by protons.
    Units of s-1 m3

    Scales with proton density
    """

    type: Literal["CP"]

    @field_validator("data")
    @classmethod
    def _validate(cls, v: AstropyQty):
        v.to("m3 / s")
        return v

    def simplify(self) -> "CPRate":
        temperature = self.temperature.to(u.K)
        data = self.data.to("m3 / s")
        return CPRate(type="CP", temperature=temperature, data=data)


class CHRate(TemperatureInterpolationRate, type_name="CH"):
    """
    Collisional excitation by neutral hydrogen.
    Units of s-1 m3

    Scales with neutral H density
    """

    type: Literal["CH"]

    @field_validator("data")
    @classmethod
    def _validate(cls, v: AstropyQty):
        v.to("m3 / s")
        return v

    def simplify(self) -> "CHRate":
        temperature = self.temperature.to(u.K)
        data = self.data.to("m3 / s")
        return CHRate(type="CH", temperature=temperature, data=data)


class ChargeExcHRate(TemperatureInterpolationRate, type_name="ChargeExcH"):
    """
    Charge exchange with neutral H. Downward rate only.
    Units of s-1 m3

    Scales with neutral H density
    """

    type: Literal["ChargeExcH"]

    @field_validator("data")
    @classmethod
    def _validate(cls, v: AstropyQty):
        v.to("m3 / s")
        return v

    def simplify(self) -> "ChargeExcHRate":
        temperature = self.temperature.to(u.K)
        data = self.data.to("m3 / s")
        return ChargeExcHRate(type="ChargeExcH", temperature=temperature, data=data)


class ChargeExcPRate(TemperatureInterpolationRate, type_name="ChargeExcP"):
    """
    Charge exchange with protons. Upward rate only.
    Units of s-1 m3

    Scales with proton density
    """

    type: Literal["ChargeExcP"]

    @field_validator("data")
    @classmethod
    def _validate(cls, v: AstropyQty):
        v.to("m3 / s")
        return v

    def simplify(self) -> "ChargeExcPRate":
        temperature = self.temperature.to(u.K)
        data = self.data.to("m3 / s")
        return ChargeExcPRate(type="ChargeExcP", temperature=temperature, data=data)


class TransCollisionalRates(CrtafBaseModel):
    """
    Container for all the rates affecting a given transition.
    """

    transition: List[str]
    data: List[SerializeAsAny[CollisionalRate]]

    @field_validator("transition")
    @classmethod
    def _validate_transition(cls, v: Any):
        length = len(v)
        if length != 2:
            raise ValueError(f"Transitions can only be between two levels, got {v}")
        return v

    def simplify(self):
        data = [d.simplify() for d in self.data]
        return TransCollisionalRates(
            transition=self.transition,
            data=data,
        )

    def format_yaml_dict(self, c: CommentedMap) -> CommentedMap:
        c["transition"].fa.set_flow_style()
        return c


class Atom(CrtafBaseModel):
    crtaf_meta: Metadata
    element: Element
    levels: Dict[str, SerializeAsAny[AtomicLevel]]
    radiative_bound_bound: List[SerializeAsAny[AtomicBoundBound]]
    radiative_bound_free: List[SerializeAsAny[AtomicBoundFree]]
    collisional_rates: List[TransCollisionalRates]

    @model_validator(mode="after")
    def _validate(self):
        # NOTE(cmo): Validate that transitions relate two named levels that
        # exist on the model.  We have already validated that each transition
        # array only contains two elements.
        def test_transition_present(name: str, type_name: str, index: int):
            if name not in self.levels:
                raise ValueError(
                    f'Transition "{name}" not found on model (seen on {type_name}[{index}]).'
                )

        energy_eV = {
            k: v.energy.to(u.eV, equivalencies=u.spectral())
            for k, v in self.levels.items()
        }

        def sort_transitions(transitions: List[str]) -> List[str]:
            # NOTE(cmo): Sort transition keys to be in descending energy order (i.e. [j, i])
            return sorted(transitions, key=lambda t: energy_eV[t], reverse=True)  # type: ignore

        for i, line in enumerate(self.radiative_bound_bound):
            for j in range(2):
                test_transition_present(line.transition[j], "radiative_bound_bound", i)
            line.transition = sort_transitions(line.transition)

        for i, cont in enumerate(self.radiative_bound_free):
            for j in range(2):
                test_transition_present(cont.transition[j], "radiative_bound_free", i)
            cont.transition = sort_transitions(cont.transition)

        for i, coll in enumerate(self.collisional_rates):
            for j in range(2):
                test_transition_present(coll.transition[j], "collisional_rates", i)
            coll.transition = sort_transitions(coll.transition)
        return self

    def simplify_visit(self, visitor: AtomicSimplificationVisitor):
        root = [self]
        levels = {k: visitor.visit(v, roots=root) for k, v in self.levels.items()}
        lines = [visitor.visit(v, roots=root) for v in self.radiative_bound_bound]
        cont = [visitor.visit(v, roots=root) for v in self.radiative_bound_free]
        coll = [visitor.visit(v, roots=root) for v in self.collisional_rates]
        new_meta = Metadata(
            version=spec_version,
            level="simplified",
            extensions=list(visitor.extensions_encountered),
            notes=self.crtaf_meta.notes,
        )
        return Atom(
            crtaf_meta=new_meta,
            element=self.element,
            levels=levels,
            radiative_bound_bound=lines,
            radiative_bound_free=cont,
            collisional_rates=coll,
        )
