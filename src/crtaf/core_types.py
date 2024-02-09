from enum import Enum
from typing import Any, Callable, ClassVar, Dict, List, Literal, Optional, Union
from typing_extensions import Annotated
import numpy as np
from pydantic import BaseModel, GetCoreSchemaHandler, SerializeAsAny, StringConstraints, ValidatorFunctionWrapHandler, field_serializer, field_validator, model_serializer, model_validator
from pydantic.config import ConfigDict
import pydantic_core.core_schema as core_schema
from pydantic_numpy import NpNDArrayFp64
import astropy.units as u

class RegisterQuantitiesMixin:
    """Simple mixin to allow for keeping track of all Quantities on a type and
    mapping unit conversions. Needs to be applied to a pydantic model`"""

    def qty_children(self):
        for k, v in self:
            if isinstance(v, RegisterQuantitiesMixin):
                yield v

    def named_qty_children(self):
        for k, v in self:
            if isinstance(v, RegisterQuantitiesMixin):
                yield k, v

    def quantities(self):
        for k, v in self:
            if isinstance(v, u.Quantity):
                yield v

    def named_quantities(self):
        for k, v in self:
            if isinstance(v, u.Quantity):
                yield k, v

    def apply_unit_conversion(self, operation: Callable[[u.Quantity], u.Quantity]):
        for k, v in self.named_quantities():
            result = operation(v)
            self.__setattr__(k, result)

        for child in self.qty_children():
            child.apply_unit_conversion(operation)


class DimensionalQuantity(BaseModel):
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
            return qty.value << validated_unit

        def serialise_qty(qty: u.Quantity):
            value = qty.value
            if len(value.shape) > 0:
                value = value.tolist()
            unit = qty.unit
            return {
                "unit": unit.to_string(),
                "value": value,
            }


        from_dim_qty_schema = core_schema.chain_schema([
            DimensionalQuantity.__pydantic_core_schema__,
            core_schema.no_info_plain_validator_function(validate_qty),
        ])
        direct_qty_schema = core_schema.is_instance_schema(cls=u.Quantity)

        schema = core_schema.union_schema([
                direct_qty_schema,
                from_dim_qty_schema,
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(serialise_qty)
        )

        return schema

AstropyQty = Annotated[u.Quantity, _AstropyQtyAnnotation]

class CollisionalRate(BaseModel):
    type: str
    temperature: AstropyQty
    data: AstropyQty

class TransCollisionalRates(BaseModel):
    transition: List[str] # Needs validator
    data: List[CollisionalRate]

class FileLevel(Enum):
    HighLevel = "high-level"
    Simplified = "simplified"

class Metadata(BaseModel):
    version: Annotated[str, StringConstraints(strip_whitespace=True, pattern=r'^v\d+\.\d+(\.\d+)*')]
    level: FileLevel
    extensions: List[str]
    notes: Optional[str]

class Element(BaseModel):
    symbol: str
    atomic_mass: float
    abundance: Optional[float]
    N: Optional[int]
    Z: Optional[int]

class Fraction(BaseModel):
    numerator: int
    denominator: int

class AtomicLevel(BaseModel):
    energy: AstropyQty
    g: int
    stage: int
    label: Optional[str]
    J: Optional[Union[Fraction, float]]
    L: Optional[int]
    S: Optional[Union[Fraction, float]]
    # TODO(cmo): validation

class SimplifiedAtomicLevel(AtomicLevel):
    energy: AstropyQty
    energy_eV: AstropyQty
    g: int
    stage: int
    label: Optional[str]
    J: Optional[Union[Fraction, float]]
    L: Optional[int]
    S: Optional[Union[Fraction, float]]
    # TODO(cmo): validation

class LineBroadening(BaseModel):
    pass

class WavelengthGrid(BaseModel):
    pass

class AtomicBoundBound(BaseModel):
    type: str
    transition: List[str] # TODO val
    f_value: float
    broadening: List[LineBroadening]
    wavelength_grid: WavelengthGrid

class SimplifiedAtomicBoundBound(AtomicBoundBound):
    type: str
    transition: List[str] # TODO val
    f_value: float
    broadening: List[LineBroadening]
    wavelength_grid: WavelengthGrid
    Aji: AstropyQty
    Bji: AstropyQty
    Bji_wavelength: AstropyQty
    Bij: AstropyQty
    Bij_wavelength: AstropyQty
    lambda0: AstropyQty

class AtomicBoundFree(RegisterQuantitiesMixin, BaseModel):
    _is_polymorphic_base: ClassVar = True
    _registry: ClassVar = {}

    type: str
    transition: List[str]

    def __init_subclass__(cls, rate_name: str, is_polymorphic_base: bool = False, **kwargs: ConfigDict):
        cls._registry[rate_name] = cls
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
            raise ValueError(f"Type {t} is not in the known types for AtomicBoundFree ({cls._registry.keys()})")

        return cls._registry[t].model_validate(v)

    @field_validator("transition")
    @classmethod
    def _validate_transition(cls, v: Any):
        length = len(v)
        if length != 2:
            raise ValueError(f"Transitions can only be between two levels, got {v}")
        return v


AtomicBoundFreeImpl = SerializeAsAny[AtomicBoundFree]

class HydrogenicBoundFree(AtomicBoundFreeImpl, rate_name="Hydrogenic"):
    type: Literal["Hydrogenic"]
    sigma_peak: AstropyQty
    lambda_min: AstropyQty
    n_lambda: int

class TabulatedBoundFreeIntermediate(BaseModel):
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
            raise ValueError(f"Expected at least 2 rows of (wavelength, cross-section), got {v.shape[0]}.")
        return v

class TabulatedBoundFree(AtomicBoundFreeImpl, rate_name="Tabulated"):
    type: Literal["Tabulated"]
    wavelengths: AstropyQty
    sigma: AstropyQty

    @model_validator(mode='wrap')
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
        assert wavelength_unit.is_equivalent(u.m, equivalencies=u.spectral()), "Wavelength unit (first unit) is not convertible to length."
        sigma_unit = u.Unit(intermediate.unit[1])
        assert sigma_unit.is_equivalent('m2'), "Cross-section unit (second unit) is not convertible to area."
        wavelengths = np.ascontiguousarray(intermediate.value[:, 0])
        sigma = np.ascontiguousarray(intermediate.value[:, 1])

        return handler({
            "type": intermediate.type,
            "transition": intermediate.transition,
            "wavelengths": wavelengths << wavelength_unit,
            "sigma": sigma << sigma_unit,
        })

    @model_serializer
    def _serialise(self):
        value = np.hstack((self.wavelengths.value[:, None], self.sigma.value[:, None]))
        value = value.tolist()
        unit = [self.wavelengths.unit.to_string(), self.sigma.unit.to_string()]
        return {
            "type": self.type,
            "transition": self.transition,
            "unit": unit,
            "value": value,
        }


class BfHolder(BaseModel):
    bf: AtomicBoundFreeImpl

class Atom(BaseModel):
    meta: Metadata
    levels: List[AtomicLevel]
    radiative_bound_bound: List[AtomicBoundBound]
    collisional_rates: List[TransCollisionalRates]
