import pydantic
import pytest
from crtaf.core_types import Element


def test_normal():
    data = {
        "symbol": "H",
        "atomic_mass": 1.005,
        "abundance": 12.0,
    }

    Element.model_validate(data)


def test_missing():
    data = {
        "atomic_mass": 1.005,
        "abundance": 12.0,
    }

    with pytest.raises(pydantic.ValidationError):
        Element.model_validate(data)


def test_greater_than_H_abundance():
    data = {
        "symbol": "H",
        "atomic_mass": 1.005,
        "abundance": 13.0,
    }

    with pytest.raises(pydantic.ValidationError):
        Element.model_validate(data)
