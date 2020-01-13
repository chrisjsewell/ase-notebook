"""Tests for ``ase_notebook.gui``."""
from ase import Atoms
import numpy as np
import pytest

from ase_notebook import draw_utils


def test_get_cell_coordinates():
    """Test coordinates of the cell lines are computed correctly."""
    line_starts, line_ends = draw_utils.get_cell_coordinates(
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
    line_starts, line_ends = draw_utils.get_cell_coordinates(
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
    points = draw_utils.get_miller_coordinates(
        [[1, 0, 0], [0, 2, 0], [0, 0, 3]], miller
    )
    assert points.tolist() == expected


def test_compute_bonds():
    """Test bonds are correctly computed (no PBC)."""
    atoms = Atoms(
        numbers=[1, 2, 3],
        positions=[[0, 0, 0], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5]],
        cell=np.eye(3),
        pbc=False,
    )
    atom_radii = np.array([0.4, 0.4, 0.4])

    bonds = draw_utils.compute_bonds(atoms, atom_radii)
    # print(bonds.tolist())
    assert np.allclose(bonds, [[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [1, 2, 0, 0, 0]])


def test_compute_bonds_pbc():
    """Test bonds are correctly computed (with PBC)."""
    atoms = Atoms(numbers=[1], positions=[[0, 0, 0]], cell=np.eye(3), pbc=True)
    atom_radii = np.array([0.4])

    bonds = draw_utils.compute_bonds(atoms, atom_radii)
    # print(bonds.tolist())
    assert np.allclose(
        bonds,
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, -1],
            [0, 0, 0, -1, 0],
            [0, 0, -1, 0, 0],
        ],
    )


def test_compute_bonds_filter_by_index():
    """Test filtering of bonds by atom indices."""
    atoms = Atoms(
        numbers=[1, 2], positions=[[0, 0, 0], [0.5, 0.5, 0.5]], cell=np.eye(3), pbc=True
    )
    # atoms.set_array("bonds", np.array([False, True]))
    atom_radii = np.array([0.4, 0.4])
    bonds = draw_utils.compute_bonds(atoms, atom_radii)
    bonds = draw_utils.filter_bond_indices(bonds, [False, True])

    # print(bonds.tolist())
    assert np.allclose(
        bonds,
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 0, -1],
            [1, 1, 0, -1, 0],
            [1, 1, -1, 0, 0],
        ],
    )


def test_compute_bonds_filter_element_pairs():
    """Test filtering of bonds by element pairs."""
    atoms = Atoms(
        symbols=["H", "He", "Mg"],
        positions=[[0, 0, 0], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5]],
        cell=np.eye(3),
        pbc=False,
    )
    atom_radii = np.array([0.4, 0.4, 0.4])

    bonds = draw_utils.compute_bonds(atoms, atom_radii)

    allowed = [("H", "Mg")]
    allowed = set([(a, b) for a, b in allowed] + [(b, a) for a, b in allowed])
    symbols = atoms.get_chemical_symbols()
    bonds = bonds[[(symbols[i], symbols[j]) in allowed for i, j in bonds[:, 0:2]]]

    # print(bonds.tolist())
    assert np.allclose(bonds, [[0, 2, 0, 0, 0]])
