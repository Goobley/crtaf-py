from copy import deepcopy
import ruamel.yaml
from crtaf.simplification_visitors import default_visitors
from crtaf.core_types import Atom, AtomicSimplificationVisitor, CollisionalRate, NaturalBroadening, OmegaRate
from io import StringIO
import astropy.units as u
import numpy as np

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
                },
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
                }
            ],
            "wavelength_grid": {
                "type": "Linear",
                "n_lambda": 201,
                "delta_lambda": 0.01 * u.nm,
            }
        },
        {
            "type": "Voigt",
            "transition": ["second", "first"],
            "f_value": 0.123,
            "broadening": [
                {
                    "type": "Natural",
                    "value": {
                        "value": 1e9,
                        "unit": "1 / s"
                    }
                },
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
                }
            ],
            "wavelength_grid": {
                "type": "Tabulated",
                "wavelengths": {
                    "unit": "Angstrom",
                    "value": [-10, 0, 5, 10],
                }
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
        {
            "transition": ["111third", "first"],
            "data": [
                {
                    "type": "CI",
                    "temperature": {
                        "unit": "K",
                        "value": [1000, 2000],
                    },
                    "data": {
                        "unit": "cm3 / (s K(1/2))",
                        "value": [50, 70]
                    }
                },
                {
                    "type": "ChargeExcP",
                    "temperature": {
                        "unit": "K",
                        "value": [1000, 2000],
                    },
                    "data": {
                        "unit": "m3 / s",
                        "value": [50, 70]
                    }
                },
            ]
        },
    ],
}
high_level_yaml = r"""meta:
  version: v0.1.0
  level: high-level
  extensions: []
  notes: A test.
element:
  symbol: Ca
  atomic_mass: 40.005
  abundance: 6.0
levels:
  first:
    energy:
      unit: 1 / cm
      value: 123.0
    g: 1
    stage: 1
    label: First Level
  second:
    energy:
      unit: 1 / cm
      value: 456.0
    g: 2
    stage: 1
    label: Second Level
  111third:
    energy:
      unit: 1 / cm
      value: 789.0
    g: 1
    stage: 2
    label: Third Level
radiative_bound_bound:
- type: PRD-Voigt
  transition: [second, first]
  f_value: 0.1
  broadening:
  - {type: Natural, elastic: false, value: {unit: 1 / s, value: 10000000.0}}
  - {type: Stark_Multiplicative, elastic: true, C_4: {unit: m3 / s, value: 7.0}, scaling: 3.0}
  - {type: Stark_Linear_Sutton, elastic: true}
  wavelength_grid:
    type: Linear
    n_lambda: 201
    delta_lambda:
      unit: nm
      value: 0.01
- type: Voigt
  transition: [second, first]
  f_value: 0.123
  broadening:
  - {type: Natural, elastic: false, value: {unit: 1 / s, value: 1000000000.0}}
  - {type: VdW_Unsold, elastic: true, H_scaling: 1.0, He_scaling: 1.5}
  - {type: Stark_Quadratic, elastic: true, scaling: 1.0}
  - {type: Stark_Linear_Sutton, elastic: true, n_upper: 3, n_lower: 2}
  wavelength_grid:
    type: Tabulated
    wavelengths: {unit: Angstrom, value: [-10.0, 0.0, 5.0, 10.0]}
radiative_bound_free:
- type: Tabulated
  transition: [111third, first]
  unit:
  - nm
  - cm2
  value:
  - [0.0, 1.0]
  - [5.2631578947368425, 2.0]
  - [10.526315789473685, 3.0]
  - [15.789473684210527, 4.0]
  - [21.05263157894737, 1.0]
  - [26.315789473684212, 2.0]
  - [31.578947368421055, 3.0]
  - [36.8421052631579, 4.0]
  - [42.10526315789474, 1.0]
  - [47.36842105263158, 2.0]
  - [52.631578947368425, 3.0]
  - [57.89473684210527, 4.0]
  - [63.15789473684211, 1.0]
  - [68.42105263157896, 2.0]
  - [73.6842105263158, 3.0]
  - [78.94736842105263, 4.0]
  - [84.21052631578948, 1.0]
  - [89.47368421052633, 2.0]
  - [94.73684210526316, 3.0]
  - [100.0, 4.0]
- type: Hydrogenic
  transition: [111third, second]
  sigma_peak:
    unit: barn
    value: 120.0
  lambda_min:
    unit: nm
    value: 45.0
  n_lambda: 40
collisional_rates:
- transition: [second, first]
  data:
  - type: Omega
    temperature:
      unit: K
      value: [10.0, 20.0, 30.0, 40.0]
    data:
      unit: ''
      value: [1.0, 2.0, 3.0, 4.0]
- transition: [111third, first]
  data:
  - type: CI
    temperature:
      unit: K
      value: [1000.0, 2000.0]
    data:
      unit: cm3 / (K(1/2) s)
      value: [50.0, 70.0]
  - type: ChargeExcP
    temperature:
      unit: K
      value: [1000.0, 2000.0]
    data:
      unit: m3 / s
      value: [50.0, 70.0]
"""

low_level_yaml = r"""meta:
  version: v0.1.0
  level: high-level
  extensions: []
  notes: A test.
element:
  symbol: Ca
  atomic_mass: 40.005
  abundance: 6.0
levels:
  first:
    energy:
      unit: 1 / cm
      value: 123.0
    g: 1
    stage: 1
    label: First Level
    energy_eV:
      unit: eV
      value: 0.015250056407283632
  second:
    energy:
      unit: 1 / cm
      value: 456.0
    g: 2
    stage: 1
    label: Second Level
    energy_eV:
      unit: eV
      value: 0.056536794485539325
  111third:
    energy:
      unit: 1 / cm
      value: 789.0
    g: 1
    stage: 2
    label: Third Level
    energy_eV:
      unit: eV
      value: 0.09782353256379502
radiative_bound_bound:
- type: PRD-Voigt
  transition: [second, first]
  f_value: 0.1
  broadening:
  - {type: Natural, elastic: false, value: {unit: 1 / s, value: 10000000.0}}
  - {type: Scaled_Exponents, elastic: true, scaling: 21.0, temperature_exponent: 0.0,
    hydrogen_exponent: 0.0, electron_exponent: 1.0}
  - {type: Scaled_Exponents, elastic: true, scaling: 0.0, temperature_exponent: 0.0,
    hydrogen_exponent: 0.0, electron_exponent: 0.6666666666666666}
  wavelength_grid:
    type: Linear
    n_lambda: 201
    delta_lambda:
      unit: nm
      value: 0.01
  Aji:
    unit: 1 / s
    value: 14793.150991996721
  Bji:
    unit: m2 / (J s)
    value: 1008373122939940.2
  Bji_wavelength:
    unit: m3 / J
    value: 0.00303327713637598
  Bij:
    unit: m2 / (J s)
    value: 504186561469970.1
  Bij_wavelength:
    unit: m3 / J
    value: 0.00151663856818799
  lambda0:
    unit: nm
    value: 30030.030030030026
- type: Voigt
  transition: [second, first]
  f_value: 0.123
  broadening:
  - {type: Natural, elastic: false, value: {unit: 1 / s, value: 1000000000.0}}
  - {type: Scaled_Exponents, elastic: true, scaling: 4.085814836572426e-14, temperature_exponent: 0.3,
    hydrogen_exponent: 1.0, electron_exponent: 0.0}
  - {type: Scaled_Exponents, elastic: true, scaling: 1.4396384663991569e-09, temperature_exponent: 0.16666666666666666,
    hydrogen_exponent: 0.0, electron_exponent: 1.0}
  - {type: Scaled_Exponents, elastic: true, scaling: 0.00102862026663837, temperature_exponent: 0.0,
    hydrogen_exponent: 0.0, electron_exponent: 0.6666666666666666}
  wavelength_grid:
    type: Tabulated
    wavelengths: {unit: nm, value: [-1.0, 0.0, 0.5, 1.0]}
  Aji:
    unit: 1 / s
    value: 18195.575720155968
  Bji:
    unit: m2 / (J s)
    value: 1240298941216126.5
  Bji_wavelength:
    unit: m3 / J
    value: 0.0037309308777424555
  Bij:
    unit: m2 / (J s)
    value: 620149470608063.2
  Bij_wavelength:
    unit: m3 / J
    value: 0.0018654654388712277
  lambda0:
    unit: nm
    value: 30030.030030030026
radiative_bound_free:
- type: Tabulated
  transition: [111third, first]
  unit:
  - nm
  - m2
  value:
  - [0.0, 0.0001]
  - [5.2631578947368425, 0.0002]
  - [10.526315789473685, 0.00030000000000000003]
  - [15.789473684210527, 0.0004]
  - [21.05263157894737, 0.0001]
  - [26.315789473684212, 0.0002]
  - [31.578947368421055, 0.00030000000000000003]
  - [36.8421052631579, 0.0004]
  - [42.10526315789474, 0.0001]
  - [47.36842105263158, 0.0002]
  - [52.631578947368425, 0.00030000000000000003]
  - [57.89473684210527, 0.0004]
  - [63.15789473684211, 0.0001]
  - [68.42105263157896, 0.0002]
  - [73.6842105263158, 0.00030000000000000003]
  - [78.94736842105263, 0.0004]
  - [84.21052631578948, 0.0001]
  - [89.47368421052633, 0.0002]
  - [94.73684210526316, 0.00030000000000000003]
  - [100.0, 0.0004]
- type: Hydrogenic
  transition: [111third, second]
  sigma_peak:
    unit: m2
    value: 1.1999999999999999e-26
  lambda_min:
    unit: nm
    value: 45.0
  n_lambda: 40
collisional_rates:
- transition: [second, first]
  data:
  - type: Omega
    temperature:
      unit: K
      value: [10.0, 20.0, 30.0, 40.0]
    data:
      unit: ''
      value: [1.0, 2.0, 3.0, 4.0]
- transition: [111third, first]
  data:
  - type: CI
    temperature:
      unit: K
      value: [1000.0, 2000.0]
    data:
      unit: m3 / (K(1/2) s)
      value: [5.000000000000001e-05, 7.000000000000001e-05]
  - type: ChargeExcP
    temperature:
      unit: K
      value: [1000.0, 2000.0]
    data:
      unit: m3 / s
      value: [50.0, 70.0]
"""

def test_yaml_regression():
    data = deepcopy(Data)
    yaml = ruamel.yaml.YAML(typ='rt')
    atom = Atom.model_validate(data)
    d = atom.yaml_dict()

    out_stream = StringIO()
    yaml.dump(d, out_stream)
    result = out_stream.getvalue()
    assert result == high_level_yaml
    assert atom.yaml_dumps() == high_level_yaml

    visitor = AtomicSimplificationVisitor(default_visitors())
    simplified = atom.simplify_visit(visitor)
    simplified_d = simplified.yaml_dict()
    out_stream = StringIO()
    yaml.dump(simplified_d, out_stream)
    result = out_stream.getvalue()
    assert result == low_level_yaml
    assert simplified.yaml_dumps() == low_level_yaml

def test_yaml_partial():
    data = deepcopy(Data)
    yaml = ruamel.yaml.YAML(typ='rt')
    atom = Atom.model_validate(data)
    d = atom.collisional_rates[0].yaml_dict()

    out_stream = StringIO()
    yaml.dump(d, out_stream)
    result = out_stream.getvalue()
    assert result == r"""transition: [second, first]
data:
- type: Omega
  temperature:
    unit: K
    value: [10.0, 20.0, 30.0, 40.0]
  data:
    unit: ''
    value: [1.0, 2.0, 3.0, 4.0]
"""

def test_yaml_load():
    rate_str = r"""type: Omega
temperature:
  unit: K
  value: [10.0, 20.0, 30.0, 40.0]
data:
  unit: ''
  value: [1.0, 2.0, 3.0, 4.0]
"""
    rate = CollisionalRate.yaml_loads(rate_str)
    assert isinstance(rate, OmegaRate)
    assert rate.temperature.unit == u.K
    assert rate.data.value[1] == 2.0

    atom = Atom.yaml_loads(high_level_yaml)
    assert isinstance(atom.radiative_bound_bound[0].broadening[0], NaturalBroadening)
