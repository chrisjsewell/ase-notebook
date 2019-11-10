"""Contains pytest fixtures."""
import os

import pytest

from ase_notebook.data import get_example_atoms


@pytest.fixture(scope="function")
def get_test_filepath():
    """Fixture to return a path to a file in the raw files folder."""
    dirpath = os.path.abspath(os.path.dirname(__file__))

    def _get_test_filepath(*path):
        return os.path.join(dirpath, "raw_files", *path)

    return _get_test_filepath


@pytest.fixture(scope="function")
def get_test_atoms():
    """Fixture to return an ase.Atoms instance by name."""
    return get_example_atoms
