
import astropy.units as u
import pytest
from crtaf.simplification_visitors import default_visitors
import numpy as np

from copy import deepcopy

from crtaf.core_types import Atom, AtomicSimplificationVisitor


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
            "energy": 0.0 * u.cm**-1,
            "g": 2,
            "stage": 2,
            "label": "First Level",
        },
        "second": {
            "energy": 25191.510 * u.cm**-1,
            "g": 2,
            "stage": 2,
            "label": "Second Level",
        },
        "111third": {
            "energy": 789.0 * u.cm**-1,
            "g": 1,
            "stage": 3,
            "label": "Third Level",
        },
    },
    "radiative_bound_bound": [
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
                "type": "LinearCoreExpWings",
                "n_lambda": 5,
                "q_core": 30.0,
                "q_wing": 1500.0,
                "vmicro_char": {
                    "value": 3.0,
                    "unit": "km / s"
                }
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
    "radiative_bound_free": [
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
    "collisional_rates": [
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

def test_linear_core_exp_wings_simplification():
    data = deepcopy(Data)
    atom = Atom.model_validate(data)
    visitor = AtomicSimplificationVisitor(default_visitors())
    simplified = atom.simplify_visit(visitor)
    assert simplified.radiative_bound_bound[0].wavelength_grid.type == "Tabulated"
    assert simplified.radiative_bound_bound[0].wavelength_grid.wavelengths.unit == u.nm
    # NOTE(cmo): Values from Lw implementation, based on default Ca II H with 5 pts.
    assert simplified.radiative_bound_bound[0].wavelength_grid.wavelengths.value == pytest.approx([-5.95850915, -0.11917018,  0.        ,  0.11917018,  5.95850915])