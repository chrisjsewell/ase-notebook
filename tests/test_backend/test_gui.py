"""Tests for ``ase_notebook.gui``."""
import json

import pytest

from ase_notebook import viewer
from ase_notebook.atoms_convert import serialize_atoms


@pytest.mark.launches_tkinter
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


@pytest.mark.launches_tkinter
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


def test_launch_gui_exec(get_test_atoms):
    """Test making a GUI, via the command-line entry point."""
    atoms = get_test_atoms("pyrite")
    ase_view = viewer.AseView()
    data_str = json.dumps(
        {
            "atoms": serialize_atoms(atoms),
            "config": ase_view.get_config_as_dict(),
            "kwargs": {"launch": False},
        }
    )
    gui = viewer.launch_gui_exec(data_str)
    try:
        pass
    finally:
        gui.exit()
