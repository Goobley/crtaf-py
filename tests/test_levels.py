from typing import Dict
import astropy.units as u
import pydantic
import pytest
from crtaf.core_types import AtomicLevel, SimplifiedAtomicLevel

def test_normal():
    data = {
        "energy": {
            "unit": "cm-1",
            "value": 123.0,
        },
        "g": 2,
        "stage": 1,
        "label": "Some Level",
    }

    AtomicLevel.model_validate(data)

def test_wrong_unit():
    data = {
        "energy": {
            "unit": "cm-2",
            "value": 123.0,
        },
        "g": 2,
        "stage": 1,
        "label": "Some Level",
    }

    with pytest.raises(pydantic.ValidationError):
        AtomicLevel.model_validate(data)

def test_JLS():
    data = {
        "energy": 123.0 * u.cm**-1,
        "g": 2,
        "stage": 1,
        "label": "Some Level",
        "J": {
            "numerator": 1,
            "denominator": 2,
        },
        "L": 0,
        "S": 0.5,
    }

    AtomicLevel.model_validate(data)

def test_J_only():
    data = {
        "energy": {
            "unit": "cm-1",
            "value": 123.0,
        },
        "g": 2,
        "stage": 1,
        "J": {
            "numerator": 1,
            "denominator": 2,
        },
    }

    with pytest.raises(pydantic.ValidationError):
        AtomicLevel.model_validate(data)

def test_simplified():
    data = {
        "energy": 10 * u.cm**-1,
        "energy_eV": 10 * u.cm**-1,
        "g": 2,
        "stage": 1,
    }

    s = SimplifiedAtomicLevel.model_validate(data)
    assert s.energy.unit == u.cm**-1
    assert s.energy_eV.unit == u.eV

def test_simplified_J_only():
    data = {
        "energy": 10 * u.cm**-1,
        "energy_eV": 10 * u.cm**-1,
        "g": 2,
        "stage": 1,
        "J": 2,
    }

    with pytest.raises(pydantic.ValidationError):
        s = SimplifiedAtomicLevel.model_validate(data)

def test_simplify_model():
    data = {
        "energy": {
            "unit": "eV",
            "value": 3.0,
        },
        "g": 2,
        "stage": 1,
        "label": "Some Level",
    }

    l = AtomicLevel.model_validate(data)
    s = l.simplify()
    assert s.energy.unit == u.cm**-1
    assert s.energy_eV.unit == u.eV
    assert s.g == 2
    assert s.stage == 1
    assert s.label == "Some Level"

def test_parse_levels(): 
    class Levels(pydantic.BaseModel):
        levels: Dict[str, pydantic.SerializeAsAny[AtomicLevel]]

    data = {
        "levels": {
            "first": {
                "energy": 123.0 * u.cm**-1,
                "g": 1,
                "stage": 1,
                "label": "First Level",
            },
            "second": {
                "energy": 456.0 * u.cm**-1,
                "g": 2,
                "stage": 1,
                "label": "Second Level",
            },
            "111third": {
                "energy": 789.0 * u.cm**-1,
                "g": 1,
                "stage": 2,
                "label": "Third Level",
            },
        },
    }

    l = Levels.model_validate(data)
    assert l.levels["111third"].energy == 789.0 * u.cm**-1
    new_l = Levels.model_validate(l.model_dump())
    for name, level in new_l.levels.items():
        new_l.levels[name] = level.simplify()

    assert new_l.levels["first"].energy_eV.unit == u.eV
