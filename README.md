crtaf
=====
Python utilities for the [Common Radiative Transfer Atomic Format](https://github.com/Goobley/CommonRTAtomicFormat).
-------------------------------------------------------------------------------------------------------------------


|   |   |   |   |
|---|---|---|---|
| __Maintainer__ | Chris Osborne | __Institution__ | University of Glasgow  |
| __License__ | ![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue) | __CI__ | ![https://github.com/Goobley/crtaf-py/actions/workflows/run_tests.yaml/badge.svg?branch=main](https://github.com/Goobley/crtaf-py/actions/workflows/run_tests.yaml/badge.svg?branch=main)


Description
-----------

`crtaf` is a simple package for managing and manipulating atomic data for radiative transfer (RT) codes, in the CRTAF YAML format.
This format is documented [here](https://github.com/Goobley/CommonRTAtomicFormat).
It is a two-tier format with both a high-level and a simplified representation (to reduce the level of atomic physics necessary to be incorporated in new RT codes), and can be easily added to most codes with the addition of a standard YAML parser.

`crtaf` allows for the simplification of high-level (intended to be writted by humans) to low-level formats.
This format supports a common set of functionality present in non-LTE radiative transfer codes such as [Lightweaver](https://github.com/Goobley/Lightweaver), RH, SNAPI.

This package is primarily of use to those working with atomic data to feed into radiative transfer models.

📖 Documentation
----------------

The package is documented via docstrings, and the [format specification](https://github.com/Goobley/CommonRTAtomicFormat).

⬇ Installation
--------------

The package should install through a normal clone and `pip install .` procedure.
It will be added to PyPI shortly.

🤝 Contributing
---------------

We would love for you to get involved.
Please report any bugs encountered on the GitHub issue tracker.

Adding features:
- For new features that don't affect the specification, please submit a pull
request directly or discuss in the issues.
- For features that affect the specification, please open an issue/pull request there first, indicating the necessary format changes/extensions. The implementation changes can then be submitted here.

We require all contributors to abide by the [code of conduct](CODE_OF_CONDUCT.md).

Acknowledgments
---------------

This format is based on the work of Tiago Pereira for [Muspel.jl](https://github.com/tiagopereira/Muspel.jl), along with the atoms as code approach of [Lightweaver](https://github.com/Lightweaver)
