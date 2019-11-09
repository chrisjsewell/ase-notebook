"""Tests for ``ase_notebook.gui``."""
import os
import sys

import pytest

from ase_notebook import viewer


@pytest.mark.skipif(
    sys.platform != "darwin" and not os.environ.get("DISPLAY", False),
    reason="no display available for gui",
)
def test_make_gui(get_test_atoms):
    """Test ``make_gui``, from the viewer."""
    atoms = get_test_atoms("pyrite")
    ase_view = viewer.AseView(
        miller_planes=[
            {"h": 1, "k": 1, "l": 0, "fill_color": "blue", "stroke_width": 1},
            {"h": 1, "k": 2, "l": 1, "fill_color": "green", "stroke_width": 1},
        ],
        show_bonds=True,
    )
    gui = ase_view.make_gui(atoms, launch=False)
    try:
        pass
    finally:
        gui.exit()


@pytest.mark.skipif(
    sys.platform != "darwin" and not os.environ.get("DISPLAY", False),
    reason="no display available for gui",
)
def test_make_gui_occupancies(get_test_atoms):
    """Test making a GUI, with atoms that contain occupancies."""
    atoms = get_test_atoms("pyrite")
    atoms.info["occupancy"] = {0: {"Fe": 0.75, "S": 0.1, "H": 0.1}}
    ase_view = viewer.AseView(
        miller_planes=[
            {"h": 1, "k": 1, "l": 0, "fill_color": "blue", "stroke_width": 1},
            {"h": 1, "k": 2, "l": 1, "fill_color": "green", "stroke_width": 1},
        ],
        miller_as_lines=True,
        show_bonds=True,
    )
    gui = ase_view.make_gui(atoms, launch=False)
    try:
        pass
    finally:
        gui.exit()


# Add a test that the console script is registered
