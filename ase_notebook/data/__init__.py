"""Module for storing and loading data files."""
import importlib_resources

from ase_notebook import data
from ase_notebook.atoms_convert import deserialize_atoms


def load_data_file(name, as_binary=False):
    """Load a data file."""
    if as_binary:
        return importlib_resources.read_binary(data, name)
    return importlib_resources.read_text(data, name)


def get_example_atoms(name="pyrite"):
    """Load an example ase.Atoms instance."""
    data = load_data_file(f"example_{name}.atoms.json")
    return deserialize_atoms(data)
