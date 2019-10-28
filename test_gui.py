"""Tests for ``aiida_2d.visualize.gui``."""
import os
import sys

import pytest

from aiida_2d.visualize import viewer


@pytest.mark.skipif(
    sys.platform != "darwin" and not os.environ.get("DISPLAY", False),
    reason="no display available for gui",
)
def test_make_gui(get_test_structure):
    """Test ``view_atoms_snapshot``."""
    structure = get_test_structure("pyrite")
    ase_view = viewer.AseView(
        miller_planes=[
            {"index": [1, 1, 0], "color": "blue", "stroke_width": 1, "as_poly": False},
            {"index": [1, 2, 1], "color": "green", "stroke_width": 1, "as_poly": True},
        ],
        show_bonds=True,
    )
    gui = ase_view.make_gui(structure, launch=False)
    try:
        pass
    finally:
        gui.exit()
