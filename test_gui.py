"""Tests for ``aiida_2d.visualize.gui``."""
import os
import sys

import numpy as np
import pytest

from aiida_2d.visualize import core, gui


def test_get_cell_coordinates():
    """Test coordinates of the cell lines are computed correctly."""
    line_starts, line_ends = core.get_cell_coordinates(
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    )
    # print(line_starts.tolist())
    # print(line_ends.tolist())
    assert line_starts.tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 2.0, 0.0],
        [1.0, 0.0, 3.0],
        [1.0, 2.0, 3.0],
    ]
    assert line_ends.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 0.0, 3.0],
        [1.0, 0.0, 3.0],
        [0.0, 2.0, 3.0],
        [0.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [0.0, 2.0, 3.0],
    ]


def test_get_cell_coordinates_2d():
    """Test that cells with a zero length vector return a plane."""
    line_starts, line_ends = core.get_cell_coordinates(
        np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
    )
    # print(line_starts.tolist())
    # print(line_ends.tolist())
    assert line_starts.tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 0.0],
    ]
    assert line_ends.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]


@pytest.mark.parametrize(
    "miller,expected",
    (
        (
            [2, 0, 0],
            [[0.5, 0.0, 0.0], [0.5, 2.0, 0.0], [0.5, 2.0, 3.0], [0.5, 0.0, 3.0]],
        ),
        (
            [0, 4, 0],
            [[0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [1.0, 0.5, 3.0], [0.0, 0.5, 3.0]],
        ),
        (
            [0, 0, 3],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 2.0, 1.0], [0.0, 2.0, 1.0]],
        ),
        (
            [2, 1, 0],
            [[0.5, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 3.0], [0.5, 0.0, 3.0]],
        ),
        ([1, 1, 1], [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
    ),
)
def test_get_miller_coordinates(miller, expected):
    """Test miller intercepts of the cell are computed correctly."""
    points = core.get_miller_coordinates([[1, 0, 0], [0, 2, 0], [0, 0, 3]], miller)
    assert points.tolist() == expected


@pytest.mark.skipif(
    sys.platform != "darwin" and not os.environ.get("DISPLAY", False),
    reason="no display available for gui",
)
def test_snapshot_pil(get_test_structure):
    """Test ``view_atoms_snapshot``."""
    from PIL.EpsImagePlugin import EpsImageFile

    structure = get_test_structure("pyrite")
    image = gui.view_atoms_snapshot(
        structure,
        miller_planes=[(1, 1, 0, "blue", 1, False), (1, 1, 1, "green", 1, True)],
    )
    assert isinstance(image, EpsImageFile), image


@pytest.mark.skipif(
    sys.platform != "darwin" and not os.environ.get("DISPLAY", False),
    reason="no display available for gui",
)
def test_snapshot_ipy(get_test_structure):
    """Test ``view_atoms_snapshot``."""
    from IPython.display import SVG

    structure = get_test_structure("pyrite")
    image = gui.view_atoms_snapshot(
        structure,
        miller_planes=[(1, 1, 0, "blue", 1, False), (1, 1, 1, "green", 1, True)],
        out_format="svg_ipy",
    )
    assert isinstance(image, SVG), image
