from crtaf.core_types import HydrogenicBoundFree
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
