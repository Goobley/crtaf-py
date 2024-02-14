import pydantic
from crtaf.core_types import ChargeExcPRate, CollisionalRate, OmegaRate
import pytest
import astropy.units as u

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

def test_different_lengths():
    rate_dict = {
        "type": "Omega",
        "temperature": {
            "unit": "K",
            "value": [1, 2, 3, 4],
        },
        "data": {
            "unit": "m/m",
            "value": [4, 3, 2, 1, 0],
        }
    }

    with pytest.raises(pydantic.ValidationError):
        rate = CollisionalRate.model_validate(rate_dict)

def test_CE():
    rate_dict = {
        "type": "CE",
        "temperature": {
            "unit": "K",
            "value": [1000, 2000, 3000, 4000],
        },
        "data": {
            "unit": "cm3 / (s K(1/2))",
            "value": [1, 2, 3, 4],
        }
    }
    rate = CollisionalRate.model_validate(rate_dict)
    assert rate.type == "CE"
    r2 = rate.simplify()
    assert r2.data.unit == u.Unit("m3 / (s K(1/2))")

def test_CI():
    rate_dict = {
        "type": "CI",
        "temperature": {
            "unit": "K",
            "value": [1000, 2000, 3000, 4000],
        },
        "data": {
            "unit": "m3 / (h K(1/2))",
            "value": [1, 2, 3, 4],
        }
    }
    rate = CollisionalRate.model_validate(rate_dict)
    assert rate.type == "CI"
    r2 = rate.simplify()
    assert r2.data.unit == u.Unit("m3 / (s K(1/2))")

def test_CP():
    rate_dict = {
        "type": "CP",
        "temperature": {
            "unit": "K",
            "value": [1000, 2000, 3000, 4000],
        },
        "data": {
            "unit": "m3 / (h K(1/2))",
            "value": [1, 2, 3, 4],
        }
    }
    with pytest.raises(pydantic.ValidationError):
        rate = CollisionalRate.model_validate(rate_dict)

    rate_dict["data"]["unit"] = "m3 / s"
    rate = CollisionalRate.model_validate(rate_dict)

def test_CH():
    rate_dict = {
        "type": "CP",
        "temperature": {
            "unit": "K",
            "value": 1000,
        },
        "data": {
            "unit": "cm3 / s",
            "value": 1,
        }
    }
    with pytest.raises(pydantic.ValidationError):
        rate = CollisionalRate.model_validate(rate_dict)

    rate_dict["temperature"]["value"] = [1000]
    rate_dict["data"]["value"] = [1]

    with pytest.raises(pydantic.ValidationError):
        rate = CollisionalRate.model_validate(rate_dict)

    rate_dict["temperature"]["value"] = [1000, 2000]
    rate_dict["data"]["value"] = [1, 7]

    rate = CollisionalRate.model_validate(rate_dict)
    r2 = rate.simplify()
    assert r2.data.unit == u.Unit("m3 /s")
    assert r2.data[1].value == pytest.approx(7e-6)

def test_ChargeExcH():
    rate_dict = {
        "type": "ChargeExcNeutralH",
        "temperature": {
            "unit": "K",
            "value": [1000, 2000, 3000, 4000],
        },
        "data": {
            "unit": "m3 / s",
            "value": [1, 2, 3, 4],
        }
    }
    with pytest.raises(pydantic.ValidationError):
        rate = CollisionalRate.model_validate(rate_dict)

    rate_dict["type"] = "ChargeExcH"
    rate = CollisionalRate.model_validate(rate_dict)

def test_ChargeExcP():
    rate_dict = {
        "type": "ChargeExcP",
        "temperature": {
            "unit": "K",
            "value": [[1000, 2000, 3000, 4000], [10000, 20000, 30000, 40000]],
        },
        "data": {
            "unit": "m3 / s",
            "value": [[1, 2, 3, 4], [5, 6, 7, 8]],
        }
    }
    with pytest.raises(pydantic.ValidationError):
        rate = CollisionalRate.model_validate(rate_dict)

    rate_dict["temperature"]["value"] = rate_dict["temperature"]["value"][1]
    rate_dict["data"]["value"] = rate_dict["data"]["value"][1]
    rate = CollisionalRate.model_validate(rate_dict)
    assert isinstance(rate, ChargeExcPRate)


