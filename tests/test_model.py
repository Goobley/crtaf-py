from copy import deepcopy
import numpy as np
import pydantic
import pytest
from crtaf.core_types import Atom, AtomicSimplificationVisitor, ScaledExponents
import astropy.units as u

from crtaf.simplification_visitors import default_visitors

Data = {
    "crtaf_meta": {
        "version": "v0.1.0",
        "level": "high-level",
        "extensions": [],
        "notes": "A test.",
    },
    "element": {
        "symbol": "Ca",
        "atomic_mass": 40.005,
        "abundance": 6.0,
    },
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
    "lines": [
        {
            "type": "PRD-Voigt",
            "transition": ["second", "first"],
            "f_value": 0.1,
            "broadening": [
                {"type": "Natural", "value": {"value": 1e7, "unit": "1 / s"}},
                {
                    "type": "Stark_Multiplicative",
                    "C_4": {
                        "unit": "m3 / s",
                        "value": 7.0,
                    },
                    "scaling": 3.0,
                },
                {
                    "type": "Stark_Linear_Sutton",
                },
            ],
            "wavelength_grid": {
                "type": "Linear",
                "n_lambda": 201,
                "delta_lambda": 0.01 * u.nm,
            },
        },
        {
            "type": "Voigt",
            "transition": ["second", "first"],
            "f_value": 0.123,
            "broadening": [
                {"type": "Natural", "value": {"value": 1e9, "unit": "1 / s"}},
                {
                    "type": "VdW_Unsold",
                    "He_scaling": 1.5,
                },
                {
                    "type": "Stark_Quadratic",
                },
                {
                    "type": "Stark_Linear_Sutton",
                    "n_upper": 3,
                    "n_lower": 2,
                },
            ],
            "wavelength_grid": {
                "type": "Tabulated",
                "wavelengths": {
                    "unit": "Angstrom",
                    "value": [-10, 0, 5, 10],
                },
            },
        },
    ],
    "continua": [
        {
            "type": "Tabulated",
            "transition": ["first", "111third"],
            "wavelengths": np.linspace(0, 100, 20) * u.nm,
            "sigma": {"unit": "cm2", "value": [1, 2, 3, 4] * 5},
        },
        {
            "type": "Hydrogenic",
            "transition": ["second", "111third"],
            "sigma_peak": 120 * u.barn,
            "lambda_min": 45.0 * u.nm,
            "n_lambda": 40,
        },
    ],
    "collisions": [
        {
            "transition": ["first", "second"],
            "data": [
                {
                    "type": "Omega",
                    "temperature": {"unit": "K", "value": [10, 20, 30, 40]},
                    "data": {"unit": "m/m", "value": [1, 2, 3, 4]},
                }
            ],
        },
        {
            "transition": ["111third", "first"],
            "data": [
                {
                    "type": "CI",
                    "temperature": {
                        "unit": "K",
                        "value": [1000, 2000],
                    },
                    "data": {"unit": "cm3 / (s K(1/2))", "value": [50, 70]},
                },
                {
                    "type": "ChargeExcP",
                    "temperature": {
                        "unit": "K",
                        "value": [1000, 2000],
                    },
                    "data": {"unit": "m3 / s", "value": [50, 70]},
                },
            ],
        },
    ],
}


def test_atom_construction():
    data = deepcopy(Data)
    a = Atom.model_validate(data)
    assert a.continua[0].transition[0] == "111third"
    assert a.collisions[0].transition[0] == "second"


def test_atom_invalid_level():
    data = deepcopy(Data)
    data["lines"][0]["transition"][0] = "2"
    with pytest.raises(pydantic.ValidationError):
        a = Atom.model_validate(data)


def test_atom_simplification():
    data = deepcopy(Data)
    visitor = AtomicSimplificationVisitor(default_visitors())

    atom = Atom.model_validate(data)
    simplified = atom.simplify_visit(visitor)
    assert simplified.continua[1].sigma.unit == u.Unit("m2")
    assert simplified.continua[1].sigma[-1].value == pytest.approx(
        atom.continua[1].sigma_peak.to("m2").value, 1e-6
    )
    assert simplified.lines[1].wavelength_grid.wavelengths.unit == u.Unit("nm")

    assert data["lines"][1]["broadening"][3]["type"] == "Stark_Linear_Sutton"
    del data["lines"][1]["broadening"][3]["n_lower"]
    with pytest.raises(pydantic.ValidationError):
        Atom.model_validate(data)
    data["lines"][1]["broadening"][3]["n_lower"] = 4
    with pytest.raises(pydantic.ValidationError):
        Atom.model_validate(data)
    data["lines"][1]["broadening"][3]["n_lower"] = 0
    with pytest.raises(pydantic.ValidationError):
        Atom.model_validate(data)

    data["lines"][1]["broadening"][3]["n_lower"] = 2
    del data["element"]["atomic_mass"]
    atom = Atom.model_validate(data)
    simplified_2 = atom.simplify_visit(visitor)

    b_orig = simplified.lines[0].broadening[1]
    b = simplified_2.lines[0].broadening[1]
    assert isinstance(b_orig, ScaledExponents)
    assert isinstance(b, ScaledExponents)
    assert b.scaling == pytest.approx(b_orig.scaling)
