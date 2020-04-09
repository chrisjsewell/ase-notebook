"""ase-notebook package setup."""
from importlib import import_module

from setuptools import find_packages, setup


setup(
    name="ase-notebook",
    version=import_module("ase_notebook").__version__,
    author="Chris Sewell",
    author_email="chrisj_sewell@hotmail.com",
    maintainer="Chris Sewell",
    maintainer_email="chrisj_sewell@hotmail.com",
    license="MIT",
    url="https://github.com/chrisjsewell/ase-notebook",
    project_urls={"Documentation": "https://ase-notebook.readthedocs.io/"},
    description=(
        "Highly configurable 2D (SVG) & 3D (threejs) visualisations "
        "for ASE/Pymatgen structures, within the Jupyter Notebook"
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.5",
    include_package_data=True,
    install_requires=[
        # core
        "ase>=3.18,<4",
        "attrs>=19,<20",
        "importlib_resources>=1,<2",
        "numpy>=1.16.4,<1.19",
        # used for color-map
        # TODO use color-map package, with no matplotlib dependency?
        "matplotlib>=3.1,<4",
        # svg
        "svgwrite>=1.3,<2",
    ],
    extras_require={
        "threejs": ["pythreejs>=2.1,<3", "ipywidgets>=7.5,<8"],
        "svgconcat": ["svgutils>=0.3,<0.4"],
        "svg2pdf": ["svglib>=0.9,<1", "reportlab>=3.5,<4"],
        "testing": [
            "coverage",
            "pytest>=3.6,<4",
            "pytest-cov",
            # "pytest-regressions",
        ],
        "code_style": [
            "black==19.3b0",
            "pre-commit==1.17.0",
            "flake8<3.8.0,>=3.7.0",
            "doc8<0.9.0,>=0.8.0",
            "pygments",  # required by doc8
        ],
        "flake8_plugins": [
            "flake8-comprehensions",
            "flake8-docstrings",
            "flake8_builtins",
            "import-order",
        ],
        "docs": ["sphinx>=1.8", "sphinx_rtd_theme", "ipypublish>=0.10.10", "ipython"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        "console_scripts": [
            "ase-notebook.view_atoms = ase_notebook.viewer:launch_gui_exec"
        ]
    },
)
