import astropy.units as u
import pytest
from crtaf.exts.linear_core_exp_wings_grid import LinearCoreExpWings
from crtaf.exts.multi_wavelength_grid import MultiWavelengthGrid
from crtaf.simplification_visitors import default_visitors
import numpy as np

from copy import deepcopy

from crtaf.core_types import Atom, AtomicSimplificationVisitor, TabulatedGrid


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
        "easy_offset": {
            "energy": 500.0 * u.nm,
            "g": 2,
            "stage": 2,
            "label": "Wavelength based",
        },
        "111third": {
            "energy": 789342.0 * u.cm**-1,
            "g": 1,
            "stage": 3,
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
                "type": "LinearCoreExpWings",
                "n_lambda": 5,
                "q_core": 30.0,
                "q_wing": 1500.0,
                "vmicro_char": {"value": 3.0, "unit": "km / s"},
            },
        },
        {
            "type": "Voigt",
            "transition": ["easy_offset", "first"],
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
                "type": "MULTI",
                "q0": 3.0,
                "q_max": 600.0,
                "n_lambda": 5,
                "q_norm": 8.0 * u.km / u.s,
            },
        },
        {
            "type": "Voigt",
            "transition": ["easy_offset", "first"],
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
                "type": "MULTI",
                "q0": 600.0,
                "q_max": 600.0,
                "n_lambda": 4,  # Will be pushed to 5
                "q_norm": 8.0 * u.km / u.s,
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


def test_ext_wavelength_simplification():
    data = deepcopy(Data)
    atom = Atom.model_validate(data)
    assert isinstance(atom.lines[0].wavelength_grid, LinearCoreExpWings)
    assert isinstance(
        atom.lines[1].wavelength_grid, MultiWavelengthGrid
    )
    visitor = AtomicSimplificationVisitor(default_visitors())
    simplified = atom.simplify_visit(visitor)

    assert simplified.lines[0].wavelength_grid.type == "Tabulated"
    assert simplified.lines[0].wavelength_grid.wavelengths.unit == u.nm
    # NOTE(cmo): Values from Lw implementation, based on default Ca II H with 5 pts.
    assert simplified.lines[
        0
    ].wavelength_grid.wavelengths.value == pytest.approx(
        [-5.95850915, -0.11917018, 0.0, 0.11917018, 5.95850915]
    )

    assert simplified.lines[1].wavelength_grid.type == "Tabulated"
    assert simplified.lines[1].wavelength_grid.wavelengths.unit == u.nm

    # NOTE(cmo): From Tiago's implementation
    multi_test = (
        np.array(
            [
                492.08474595750886,
                499.9523432985373,
                500.0,
                500.0476657878395,
                508.17405286233975,
            ]
        )
        - 500.0
    )
    assert simplified.lines[
        1
    ].wavelength_grid.wavelengths.value == pytest.approx(multi_test)
    # NOTE(cmo): Test linear case (linear in frequency, not wavelength)
    grid = simplified.lines[2].wavelength_grid.wavelengths.value
    nu_grid = ((grid + 500.0) * u.nm).to(u.Hz, equivalencies=u.spectral()).value
    dnu = nu_grid[1:] - nu_grid[:-1]
    assert dnu == pytest.approx([dnu[0]] * 4)
    assert grid[0] < 0.0
    assert grid[2] == 0.0
    assert grid[4] > 0.0


def test_ext_simplification_allowed():
    data = deepcopy(Data)
    atom = Atom.model_validate(data)
    visitor = AtomicSimplificationVisitor(
        default_visitors(), extensions=["multi_wavelength_grid"]
    )
    simplified = atom.simplify_visit(visitor)
    assert simplified.crtaf_meta.extensions[0] == "multi_wavelength_grid"
    assert isinstance(
        simplified.lines[0].wavelength_grid, TabulatedGrid
    )
    assert isinstance(
        simplified.lines[2].wavelength_grid, MultiWavelengthGrid
    )

    # NOTE(cmo): Intentional Typo
    visitor = AtomicSimplificationVisitor(
        default_visitors(), extensions=["______multi_wavelength_grid"]
    )
    simplified = atom.simplify_visit(visitor)
    assert isinstance(
        simplified.lines[1].wavelength_grid, TabulatedGrid
    )
