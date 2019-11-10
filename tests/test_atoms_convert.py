"""Tests for ``ase_notebook.atoms_convert``."""
import json

import numpy as np

from ase_notebook import atoms_convert


def test_atoms_convert(get_test_atoms):
    """Test roundtrip serialization/de-serialization."""
    atoms = get_test_atoms("pyrite")
    atoms.info = {"a": 1}
    atoms.set_array("label", np.array(["abc"] * 12))
    json_string = atoms_convert.serialize_atoms(atoms)
    # print(list(json.loads(json_string)))
    assert list(json.loads(json_string)) == [
        "description",
        "cell",
        "arrays",
        "info",
        "constraints",
        "celldisp",
        "calculator",
    ]
    new_atoms = atoms_convert.deserialize_atoms(json_string)
    assert atoms == new_atoms
