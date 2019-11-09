"""Tests for ``ase_notebook.atoms_info``."""
from ase_notebook import atom_info


def test_atoms_convert(get_test_atoms):
    """Creating atoms info."""
    atoms = get_test_atoms("pyrite")
    assert atom_info.create_info_lines(atoms, [1]) == [
        "#1 Iron (Fe)",
        "(-2.695, 0.000, -2.695) Å",
        "tag=0",
    ]
    assert atom_info.create_info_lines(atoms, [1, 2]) == [
        "Bond Distance Fe-Fe: 3.811 Å"
    ]
    assert atom_info.create_info_lines(atoms, [1, 2, 3]) == [
        "Valence Angle Fe-Fe-Fe: 60.0°, 60.0°, 60.0°"
    ]
    assert atom_info.create_info_lines(atoms, [1, 2, 3, 4])[0][:27] == (
        "Dihedral Fe → Fe → Fe → S: "
    )
    # TODO on travis getting 80.6 instead of 158.3
    assert atom_info.create_info_lines(atoms, [1, 2, 3, 4, 5]) == ["Formula: S3Fe4"]
