from copy import deepcopy
import numpy as np
from crtaf.core_types import Atom, AtomicSimplificationVisitor
import astropy.units as u

from crtaf.simplification_visitors import default_visitors

Data = {
    "meta": {
        "version": "v0.1.0",
        "level": "high-level",
        "extensions": [],
        "notes": "A test."
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
    "radiative_bound_bound": [
        {
            "type": "PRD-Voigt",
            "transition": ["second", "first"],
            "f_value": 0.1, 
            "broadening": [
                {
                    "type": "Natural",
                    "value": {
                        "value": 1e7,
                        "unit": "1 / s"
                    }
                }
            ],
            "wavelength_grid": {
                "type": "Linear",
                "n_lambda": 201,
                "delta_lambda": 0.01 * u.nm,
            }
        },
    ],
    "radiative_bound_free": [
        {
            "type": "Tabulated",
            "transition": ["first", "111third"],
            "wavelengths": np.linspace(0, 100, 20) * u.nm,
            "sigma": {
                "unit": "cm2",
                "value": [1, 2, 3, 4] * 5
            }
        },
        {
            "type": "Hydrogenic",
            "transition": ["second", "111third"],
            "sigma_peak": 120 * u.barn,
            "lambda_min": 45.0 * u.nm,
            "n_lambda": 40,
        },
    ],
    "collisional_rates": [
        {
            "transition": ["first", "second"],
            "data": [
                {
                    "type": "Omega",
                    "temperature": {
                        "unit": "K",
                        "value": [10, 20, 30, 40]
                    },
                    "data": {
                        "unit": "m/m",
                        "value": [1, 2, 3, 4]
                    }
                }
            ]
        },
    ],
}

def test_atom_construction():
    data = deepcopy(Data)
    a = Atom.model_validate(data)
    assert a.radiative_bound_free[0].transition[0] == "111third"

def test_atom_simplification():
    data = deepcopy(Data)
    visitor = AtomicSimplificationVisitor(default_visitors())

    atom = Atom.model_validate(data)
    simplified = atom.simplify_visit(visitor)
