import astropy.units as u
import numpy as np
from crtaf.core_types import AstropyQty
from pydantic import BaseModel

class Test(BaseModel):
    data: AstropyQty

def test_basic_list():
    data = {
        "data": {
            "unit": "1 / s",
            "value": [1, 2, 3, 4]
        }
    }
    t = Test.model_validate(data)
    assert t.data.value.shape[0] == 4
    assert t.data.unit == 1.0 / u.s

def test_premade_array():
    data = {
        "data": np.linspace(0, 10, 10) * u.m
    }
    t = Test.model_validate(data)
    assert t.data.value.shape[0] == 10
    assert t.data.unit == u.m

def test_single_dict():
    data = {
        "data": {
            "unit": "m",
            "value": 3.1,
        }
    }
    t = Test.model_validate(data)
    assert t.data.value.item() == 3.1
    assert t.data.unit == u.m

def test_single_premade_item():
    data = {
        "data": 0.314 / u.m,
    }
    t = Test.model_validate(data)
    assert t.data.value.item() == 0.314
    assert t.data.unit == 1.0 / u.m
