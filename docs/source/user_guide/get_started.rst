Installation
++++++++++++

|PyPI| |Conda|

To install from Conda (recommended)::

    >> conda install -c conda-forge ase-notebook

To install from PyPi::

    >> pip install ase-notebook

To install the development version::

    >> git clone https://github.com/chrisjsewell/ase-notebook .
    >> cd ase-notebook
    >> pip install -e .
    #>> pip install -e .[code_style,testing,docs] # install extras for more features


Development
+++++++++++

Testing
~~~~~~~

|Build Status| |Coverage Status|

The following will discover and run all unit test:

.. code:: shell

   >> cd ase-notebook
   >> pytest -v

Coding Style Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

The code style is tested using `flake8 <http://flake8.pycqa.org>`__,
with the configuration set in ``.flake8``, and
`black <https://github.com/ambv/black>`__.

Installing with ``ase-notebook[code_style]`` makes the
`pre-commit <https://pre-commit.com/>`__ package available, which will
ensure these tests are passed by reformatting the code and testing for
lint errors before submitting a commit. It can be set up by:

.. code:: shell

   >> cd ase-notebook
   >> pre-commit install

Optionally you can run ``black`` and ``flake8`` separately:

.. code:: shell

   >> black .
   >> flake8 .

Editors like VS Code also have automatic code reformat utilities, which
can check and adhere to this standard.

Documentation
~~~~~~~~~~~~~

The documentation can be created locally by:

.. code:: shell

   >> cd ase-notebook/docs
   >> make clean
   >> make  # or make debug

.. |PyPI| image:: https://img.shields.io/pypi/v/ase-notebook.svg
   :target: https://pypi.python.org/pypi/ase-notebook/
.. |Conda| image:: https://anaconda.org/conda-forge/ase-notebook/badges/version.svg
   :target: https://anaconda.org/conda-forge/ase-notebook
.. |Build Status| image:: https://travis-ci.org/chrisjsewell/ase-notebook.svg?branch=master
   :target: https://travis-ci.org/chrisjsewell/ase-notebook
.. |Coverage Status| image:: https://coveralls.io/repos/github/chrisjsewell/ase-notebook/badge.svg?branch=master
   :target: https://coveralls.io/github/chrisjsewell/ase-notebook?branch=master
