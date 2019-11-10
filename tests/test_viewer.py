"""Tests for ``ase_notebook.viewer``."""

from ase_notebook.viewer import AseView


def test_get_atom_colors(get_test_atoms):
    """Test retrieval of atom colors, via a colormap."""
    viewer = AseView(
        atom_color_by="value_array",
        atom_color_array="numbers",
        atom_colormap_range=(0, 100),
    )
    atoms = get_test_atoms("pyrite")
    colors = viewer.get_atom_colors(atoms)
    assert colors == [
        "#0088ff",
        "#0088ff",
        "#0088ff",
        "#0088ff",
        "#0020ff",
        "#0020ff",
        "#0020ff",
        "#0020ff",
        "#0020ff",
        "#0020ff",
        "#0020ff",
        "#0020ff",
    ]


def test_get_atom_labels(get_test_atoms):
    """Test retrieval of atom labels."""
    viewer = AseView(atom_label_by="array", atom_label_array="numbers")
    atoms = get_test_atoms("pyrite")
    labels = viewer.get_atom_labels(atoms)
    assert labels == [
        "26",
        "26",
        "26",
        "26",
        "16",
        "16",
        "16",
        "16",
        "16",
        "16",
        "16",
        "16",
    ]
