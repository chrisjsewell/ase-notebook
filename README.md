# ase-notebook

[![CI Status](https://travis-ci.org/chrisjsewell/ase-notebook.svg?branch=develop)](https://travis-ci.org/chrisjsewell/ase-notebook)
[![Coverage](https://coveralls.io/repos/github/chrisjsewell/ase-notebook/badge.svg?branch=develop)](https://coveralls.io/github/chrisjsewell/ase-notebook?branch=develop)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![RTD](https://readthedocs.org/projects/ase-notebook/badge)](http://ase-notebook.readthedocs.io/)
<!-- [![PyPI](https://img.shields.io/pypi/v/ase-notebook.svg)](https://pypi.org/project/ase-notebook)
[![Conda](https://anaconda.org/conda-forge/ase-notebook/badges/version.svg)](https://anaconda.org/conda-forge/ase-notebook) -->

A highly configurable 2D (SVG) &amp; 3D (threejs) visualisation creator for ASE/Pymatgen structures,
within the Jupyter Notebook.

![Example SVG](/docs/source/images/example_vis.svg)

## Purpose

To create atomic configuration visualisations, principally within a Jupyter Notebook.

## Contributing

Contributions are very welcome.

The following will discover and run all unit test:

```shell
>> pip install -e .[testing]
>> pytest -v
```

### Coding Style Requirements

The code style is tested using [flake8](http://flake8.pycqa.org),
with the configuration set in `.flake8`,
and code should be formatted with [black](https://github.com/ambv/black).

Installing with `ase-notebook[code_style]` makes the [pre-commit](https://pre-commit.com/)
package available, which will ensure these tests are passed by reformatting the code
and testing for lint errors before submitting a commit.
It can be setup by:

```shell
>> cd ase-notebook
>> pre-commit install
```

Optionally you can run `black` and `flake8` separately:

```shell
>> black .
>> flake8 .
```

Editors like VS Code also have automatic code reformat utilities, which can adhere to this standard.

## License

See LICENSE file

## Issues

If you encounter any problems, please [file an issue](https://github.com/chrisjsewell/ase-notebook/issues) along with a detailed description.
