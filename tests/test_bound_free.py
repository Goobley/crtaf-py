import astropy.units as u
import pydantic
import pytest

from crtaf.core_types import AtomicBoundFree, HydrogenicBoundFree, TabulatedBoundFree
from pytest import approx

def test_hydrogenic_basic():
    data = {
        "type": "Hydrogenic",
        "transition": ["1", "2"],
        "sigma_peak": {
            "unit": "m^2",
            "value": 123.4,
        },
        "lambda_min": {
            "unit": "nm",
            "value": 9
        },
        "n_lambda": 20
    }

    HydrogenicBoundFree.model_validate(data)

def test_hydrogenic_no_nlambda():
    data = {
        "type": "Hydrogenic",
        "transition": ["1", "2"],
        "sigma_peak": {
            "unit": "m^2",
            "value": 123.4,
        },
        "lambda_min": {
            "unit": "nm",
            "value": 92.5,
        },
    }

    with pytest.raises(pydantic.ValidationError):
        HydrogenicBoundFree.model_validate(data)

def test_hydrogenic_3_trans():
    data = {
        "type": "Hydrogenic",
        "transition": ["1", "2", "3"],
        "sigma_peak": {
            "unit": "m^2",
            "value": 123.4,
        },
        "lambda_min": {
            "unit": "nm",
            "value": 92.5,
        },
    }

    with pytest.raises(pydantic.ValidationError):
        HydrogenicBoundFree.model_validate(data)

def test_hydrogenic_base_type():
    data = {
        "type": "Hydrogenic",
        "transition": ["1", "2"],
        "sigma_peak": {
            "unit": "m^2",
            "value": 123.4,
        },
        "lambda_min": {
            "unit": "nm",
            "value": 92.5,
        },
        "n_lambda": 20
    }

    AtomicBoundFree.model_validate(data)

def test_hydrogenic_round_trip():
    data = {
        "type": "Hydrogenic",
        "transition": ["1", "2"],
        "sigma_peak": {
            "unit": "m^2",
            "value": 123.4,
        },
        "lambda_min": {
            "unit": "nm",
            "value": 92.5,
        },
        "n_lambda": 20
    }

    m = AtomicBoundFree.model_validate(data)
    dump = m.model_dump()
    AtomicBoundFree.model_validate(dump)

def test_hydrogenic_unit_conversion():
    data = {
        "type": "Hydrogenic",
        "transition": ["1", "2"],
        "sigma_peak": {
            "unit": "m^2",
            "value": 123.4,
        },
        "lambda_min": {
            "unit": "nm",
            "value": 92.5,
        },
        "n_lambda": 20
    }

    m = AtomicBoundFree.model_validate(data)
    def conversion(q: u.Quantity):
        if q.unit.physical_type == 'area':
            return q.to(u.cm**2)
        if q.unit.physical_type == 'length':
            return q.to(u.cm)
    m.apply_unit_conversion(conversion)

    assert m.sigma_peak.value == approx(123.4 * 100 * 100)
    assert m.lambda_min.value == approx(92.5e-7)
        

def test_hydrogenic_round_trip_json():
    data = {
        "type": "Hydrogenic",
        "transition": ["1", "2"],
        "sigma_peak": {
            "unit": "m^2",
            "value": 123.4,
        },
        "lambda_min": {
            "unit": "nm",
            "value": 92.5,
        },
        "n_lambda": 20
    }

    m = AtomicBoundFree.model_validate(data)
    dump = m.model_dump_json()
    AtomicBoundFree.model_validate_json(dump)

def test_tabulated_bound_free():
    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "m2"],
        "value": [
            [1, 123e-20],
            [2, 456e-20],
            [3, 789e-20],
        ]
    }

    AtomicBoundFree.model_validate(data)

def test_tabulated_direct():
    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "m2"],
        "value": [
            [1, 123e-20],
            [2, 456e-20],
            [3, 789e-20],
        ]
    }
    TabulatedBoundFree.model_validate(data)

def test_tabulated_round_trip():
    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "m2"],
        "value": [
            [1, 123e-20],
            [2, 456e-20],
            [3, 789e-20],
        ]
    }
    m = AtomicBoundFree.model_validate(data)
    dump = m.model_dump()
    AtomicBoundFree.model_validate(dump)

def test_tabulated_round_trip_json():
    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "m2"],
        "value": [
            [1, 123e-20],
            [2, 456e-20],
            [3, 789e-20],
        ]
    }
    m = AtomicBoundFree.model_validate(data)
    dump = m.model_dump_json()
    AtomicBoundFree.model_validate_json(dump)

def test_tabulated_bound_free_bad_unit():
    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "mmmm2"],
        "value": [
            [1, 123e-20],
            [2, 456e-20],
            [3, 789e-20],
        ]
    }

    with pytest.raises(pydantic.ValidationError):
        AtomicBoundFree.model_validate(data)

    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["Hz", "mm2"],
        "value": [
            [1, 123e-20],
            [2, 456e-20],
            [3, 789e-20],
        ]
    }

    AtomicBoundFree.model_validate(data)

    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["Pa", "mm2"],
        "value": [
            [1, 123e-20],
            [2, 456e-20],
            [3, 789e-20],
        ]
    }

    with pytest.raises(pydantic.ValidationError):
        AtomicBoundFree.model_validate(data)

    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "barn"],
        "value": [
            [1, 123e-20],
            [2, 456e-20],
            [3, 789e-20],
        ]
    }

    AtomicBoundFree.model_validate(data)

    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "m"],
        "value": [
            [1, 123e-20],
            [2, 456e-20],
            [3, 789e-20],
        ]
    }

    with pytest.raises(pydantic.ValidationError):
        AtomicBoundFree.model_validate(data)

def test_tabulated_bound_1_entry():
    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "m2"],
        "value": [
            [1, 123e-20],
        ]
    }

    with pytest.raises(pydantic.ValidationError):
        AtomicBoundFree.model_validate(data)

    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "m2"],
        "value": [
            1, 123e-20,
        ]
    }

    with pytest.raises(pydantic.ValidationError):
        AtomicBoundFree.model_validate(data)

def test_tabulated_bound_lowercase():
    data = {
        "type": "tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "m2"],
        "value": [
            [1, 123e-20],
            [2, 456e-20],
            [3, 789e-20],
        ],
    }

    with pytest.raises(pydantic.ValidationError):
        TabulatedBoundFree.model_validate(data)
    with pytest.raises(pydantic.ValidationError):
        AtomicBoundFree.model_validate(data)