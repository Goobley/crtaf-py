import pydantic
from crtaf.core_types import CollisionalRate, OmegaRate
import pytest

def test_normal():
    rate_dict = {
        "type": "Omega",
        "temperature": {
            "unit": "K",
            "value": [1, 2, 3, 4],
        },
        "data": {
            "unit": "m/m",
            "value": [4, 3, 2, 1],
        }
    }

    rate = CollisionalRate.model_validate(rate_dict)
    ser = rate.model_dump()
    round_trip = rate.model_validate(ser)
    assert isinstance(round_trip, OmegaRate)
    from_instance = CollisionalRate.model_validate(round_trip)
    assert isinstance(from_instance, OmegaRate)

def test_missing_data():
    rate_dict = {
        "type": "Omega",
        "temperature": {
            "unit": "K",
            "value": [1, 2, 3, 4],
        },
    }

    with pytest.raises(pydantic.ValidationError):
        rate = CollisionalRate.model_validate(rate_dict)
