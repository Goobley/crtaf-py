from enum import Enum
from typing import Any, ClassVar, List, Optional, Union
from typing_extensions import Annotated, Unpack
import numpy as np
from pydantic import BaseModel, Discriminator, Field, GetCoreSchemaHandler, StringConstraints, Tag, field_serializer, constr
from pydantic.config import ConfigDict
import pydantic_core.core_schema as core_schema
from pydantic_numpy import NpNDArrayFp64
import astropy.units as u

class DimensionalQuantity(BaseModel):
    unit: str
    value: Union[float, NpNDArrayFp64]

    @field_serializer("value")
    def serialize_value(self, value, _info):
        return value.tolist()

def make_astropy_qty(*, value, unit):
    validated_unit = u.Unit(unit)
    return value << validated_unit

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

class AtomicBoundFree(BaseModel):
    _registry: ClassVar = {}

    type: str
    transition: List[str] # TODO val

    def __init_subclass__(cls, rate_name: str, **kwargs: ConfigDict):
        cls._registry[rate_name] = cls
        return super().__init_subclass__(**kwargs)

    @staticmethod
    def get_discriminator_value(v: Any):
        if isinstance(v, dict):
            return v["type"]
        return getattr(v, "type")

    # @classmethod
    # def get_union_types(cls) -> Union:
    #     types = []
    #     for k, v in cls._registry.items():
    #         types.append(Annotated[v, Tag(k)])
    #     return Union[*tuple(types)]

# https://typethepipe.com/post/pydantic-discriminated-union/
# https://github.com/pydantic/pydantic/discussions/7008
# https://docs.pydantic.dev/latest/concepts/serialization/#serializing-with-duck-typing

class HydrogenicBoundFree(AtomicBoundFree, rate_name="Hydrogenic"):
    sigma_peak: AstropyQty
    lambda_min: AstropyQty
    n_lambda: int

# class BfHolder(BaseModel):
#     bf: Annotated[AtomicBoundFree.get_union_types(), Discriminator(AtomicBoundFree.get_discriminator_value)]

class Atom(BaseModel):
    meta: Metadata
    levels: List[AtomicLevel]
    radiative_bound_bound: List[AtomicBoundBound]
    collisional_rates: List[TransCollisionalRates]
