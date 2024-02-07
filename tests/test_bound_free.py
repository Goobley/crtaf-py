from crtaf.core_types import AtomicBoundFreeImpl, BfHolder, HydrogenicBoundFree, TabulatedBoundFree
import pydantic
import pytest

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
            "value": 92.5,
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

    AtomicBoundFreeImpl.model_validate(data)

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

    m = AtomicBoundFreeImpl.model_validate(data)
    dump = m.model_dump()
    AtomicBoundFreeImpl.model_validate(dump)

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

    m = AtomicBoundFreeImpl.model_validate(data)
    dump = m.model_dump_json()
    AtomicBoundFreeImpl.model_validate_json(dump)

def test_hydrogenic_bfholder():
    data = { 'bf': {
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
    }}

    BfHolder.model_validate(data)


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

    AtomicBoundFreeImpl.model_validate(data)

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
    m = AtomicBoundFreeImpl.model_validate(data)
    dump = m.model_dump()
    AtomicBoundFreeImpl.model_validate(dump)

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
    m = AtomicBoundFreeImpl.model_validate(data)
    dump = m.model_dump_json()
    AtomicBoundFreeImpl.model_validate_json(dump)

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
        AtomicBoundFreeImpl.model_validate(data)

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

    AtomicBoundFreeImpl.model_validate(data)

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
        AtomicBoundFreeImpl.model_validate(data)

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

    AtomicBoundFreeImpl.model_validate(data)

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
        AtomicBoundFreeImpl.model_validate(data)

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
        AtomicBoundFreeImpl.model_validate(data)

    data = {
        "type": "Tabulated",
        "transition": ["lower", "upper"],
        "unit": ["nm", "m2"],
        "value": [
            1, 123e-20,
        ]
    }

    with pytest.raises(pydantic.ValidationError):
        AtomicBoundFreeImpl.model_validate(data)

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
        AtomicBoundFreeImpl.model_validate(data)