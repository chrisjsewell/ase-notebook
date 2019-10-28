"""Implementation agnostics visualisation functions."""
from collections import namedtuple, OrderedDict
from collections.abc import Mapping
from itertools import product
from math import ceil, cos, radians, sin
from typing import List

import numpy as np


def rotate(rotations, init_rotation=None):
    """Convert string of format '50x,-10y,120z' to a rotation matrix.

    Note that the order of rotation matters, i.e. '50x,40z' is different
    from '40z,50x'.
    """
    if init_rotation is None:
        rotation = np.identity(3)
    else:
        rotation = init_rotation

    if rotations == "":
        return rotation

    for i, a in [
        ("xyz".index(s[-1]), radians(float(s[:-1]))) for s in rotations.split(",")
    ]:
        s = sin(a)
        c = cos(a)
        if i == 0:
            rotation = np.dot(rotation, [(1, 0, 0), (0, c, s), (0, -s, c)])
        elif i == 1:
            rotation = np.dot(rotation, [(c, 0, -s), (0, 1, 0), (s, 0, c)])
        else:
            rotation = np.dot(rotation, [(c, s, 0), (-s, c, 0), (0, 0, 1)])
    return rotation


def get_cell_coordinates(
    cell, origin=(0.0, 0.0, 0.0), show_repeats=None, segment_length=None
):
    """Get start and end points of lines segments used to draw unit cells.

    We also add an origin option, to allow for different cells to be created.
    """
    reps_a, reps_b, reps_c = show_repeats or (1, 1, 1)
    vec_a, vec_b, vec_c = cell
    has_a = np.linalg.norm(vec_a) > 1e-9
    has_b = np.linalg.norm(vec_b) > 1e-9
    has_c = np.linalg.norm(vec_c) > 1e-9
    vec_a = vec_a / reps_a
    vec_b = vec_b / reps_b
    vec_c = vec_c / reps_c

    lines = []

    for rep_a, rep_b, rep_c in product(
        *(range(1, reps_a + 1), range(1, reps_b + 1), range(1, reps_c + 1))
    ):
        rep_origin = (
            np.array(origin)
            + (rep_a - 1) * vec_a
            + (rep_b - 1) * vec_b
            + (rep_c - 1) * vec_c
        )
        if has_a:
            lines.append([rep_origin, rep_origin + vec_a])
        if has_b:
            lines.append([rep_origin, rep_origin + vec_b])
        if has_c:
            lines.append([rep_origin, rep_origin + vec_c])
        if has_a and has_b:
            lines.extend(
                [
                    [rep_origin + vec_a, rep_origin + vec_a + vec_b],
                    [rep_origin + vec_a + vec_b, rep_origin + vec_b],
                ]
            )
        if has_a and has_c:
            lines.extend(
                [
                    [rep_origin + vec_a, rep_origin + vec_a + vec_c],
                    [rep_origin + vec_c, rep_origin + vec_c + vec_a],
                ]
            )
        if has_b and has_c:
            lines.extend(
                [
                    [rep_origin + vec_b, rep_origin + vec_b + vec_c],
                    [rep_origin + vec_c, rep_origin + vec_c + vec_b],
                ]
            )
        if has_a and has_b and has_c:
            lines.extend(
                [
                    [rep_origin + vec_a + vec_b, rep_origin + vec_a + vec_b + vec_c],
                    [rep_origin + vec_c + vec_a, rep_origin + vec_c + vec_a + vec_b],
                    [rep_origin + vec_c + vec_a + vec_b, rep_origin + vec_c + vec_b],
                ]
            )

    lines = np.array(lines, dtype=float)

    if segment_length:
        # split lines into segments (to 'improve' the z-order of lines)
        new_lines = []
        for (start, end) in lines:
            length = np.linalg.norm(end - start)
            segments = int(ceil(length / segment_length))
            for i in range(segments):
                new_end = start + (end - start) * (i + 1) / segments
                new_lines.append([start, new_end])
                start = new_end
        lines = np.array(new_lines, dtype=float)

    return lines[:, 0], lines[:, 1]


def get_miller_coordinates(cell, miller):
    """Compute the points at which a miller index intercepts with a unit cell boundary."""
    vec_a, vec_b, vec_c = np.array(cell, dtype=float)
    hval, kval, lval = miller

    if hval < 0 or kval < 0 or lval < 0:
        # TODO compute negative miller intercepts
        # look at scipt in https://www.doitpoms.ac.uk/tlplib/miller_indices/printall.php
        # they appear to use a transpose
        raise NotImplementedError("h, k or l less than zero")

    h_is_zero, k_is_zero, l_is_zero = np.isclose(miller, 0)

    mod_a = np.inf if h_is_zero else vec_a / hval
    mod_b = np.inf if k_is_zero else vec_b / kval
    mod_c = np.inf if l_is_zero else vec_c / lval

    if h_is_zero and k_is_zero and l_is_zero:
        raise ValueError("h, k, l all 0")
    elif k_is_zero and l_is_zero:
        points = [mod_a, mod_a + vec_b, mod_a + vec_b + vec_c, mod_a + vec_c]
    elif h_is_zero and l_is_zero:
        points = [mod_b, mod_b + vec_a, mod_b + vec_a + vec_c, mod_b + vec_c]
    elif h_is_zero and k_is_zero:
        points = [mod_c, mod_c + vec_a, mod_c + vec_a + vec_b, mod_c + vec_b]
    elif h_is_zero:
        points = [mod_b, mod_c, mod_c + vec_a, mod_b + vec_a]
    elif k_is_zero:
        points = [mod_a, mod_c, mod_c + vec_b, mod_a + vec_b]
    elif l_is_zero:
        points = [mod_a, mod_b, mod_b + vec_c, mod_a + vec_c]
    else:
        points = [mod_a, mod_b, mod_c]
    return np.array(points)


def lighten_hexcolor(hexcolor, fraction):
    """Lighten a color (in hex format) by a fraction."""
    if fraction <= 0:
        return hexcolor
    hexcolor = hexcolor.lstrip("#")
    rgb = np.array([int(hexcolor[i : i + 2], 16) for i in (0, 2, 4)])
    white = np.array([255, 255, 255])
    rgb = rgb + (white - rgb) * fraction
    return "#{0:02X}{1:02X}{2:02X}".format(*(int(x * 255) for x in rgb))


def compute_bonds(atoms, atom_radii):
    """Compute bonds for atoms."""
    from ase.neighborlist import NeighborList

    nl = NeighborList(atom_radii * 1.5, skin=0, self_interaction=False)
    nl.update(atoms)
    nbonds = nl.nneighbors + nl.npbcneighbors

    bonds = np.empty((nbonds, 5), int)
    if nbonds == 0:
        return bonds

    n1 = 0
    for a in range(len(atoms)):
        indices, offsets = nl.get_neighbors(a)
        n2 = n1 + len(indices)
        bonds[n1:n2, 0] = a
        bonds[n1:n2, 1] = indices
        bonds[n1:n2, 2:] = offsets
        n1 = n2

    i = bonds[:n2, 2:].any(1)
    pbcbonds = bonds[:n2][i]
    bonds[n2:, 0] = pbcbonds[:, 1]
    bonds[n2:, 1] = pbcbonds[:, 0]
    bonds[n2:, 2:] = -pbcbonds[:, 2:]
    return bonds


class DrawElementsBase:
    """Abstract base class to store a set of 3D-visualisation elements."""

    etype = None

    def __init__(
        self, name, coordinates, element_properties=None, group_properties=None
    ):
        """Initialise the element group."""
        self.name = name
        self._coordinates = coordinates
        self._positions = coordinates
        self._axes = np.identity(3)
        self._offset = np.zeros(3)
        self._scale = 1
        self._el_props = element_properties or {}
        self._grp_props = group_properties or {}

    def __len__(self):
        """Return the number of elements."""
        return len(self._coordinates)

    def __getitem__(self, index):
        """Return a single element."""
        try:
            index = int(index)
        except ValueError:
            raise TypeError(f"index must be an integer: {index}")
        element = namedtuple(
            "Element", ["name", "type", "position"] + list(self._el_props.keys())
        )
        return element(
            self.name,
            self.etype,
            self._positions[index],
            *[v[index] for v in self._el_props.values()],
        )

    def unstack_coordinates(self):
        """Return a list of all coordinates in the group."""
        raise NotImplementedError

    def unstack_positions(self):
        """Return a list of all coordinates in the group."""
        raise NotImplementedError

    def update_positions(self, axes, offset, scale):
        """Update element positions, give a axes basis and centre offset."""
        raise NotImplementedError

    def get_max_zposition(self):
        """Return the maximum z-coordinate."""
        raise NotImplementedError


class DrawElementsSphere(DrawElementsBase):
    """Store a set of 3D-visualisation sphere elements."""

    etype = "sphere"

    def __init__(
        self, name, coordinates, radii, element_properties=None, group_properties=None
    ):
        """Initialise the element group."""
        coordinates = np.array(coordinates)
        if coordinates.shape == (0,):
            coordinates = np.empty((0, 3))
        super().__init__(name, coordinates, element_properties, group_properties)
        self._radii = np.array(radii)

    def __getitem__(self, index):
        """Return a single element."""
        element = namedtuple(
            "Element",
            ["name", "type", "position", "sradius"] + list(self._el_props.keys()),
        )
        return element(
            self.name,
            self.etype,
            self._positions[index],
            self.scaled_radii[index],
            *[v[index] for v in self._el_props.values()],
        )

    @property
    def scaled_radii(self):
        """Return the scaled radii, for each sphere."""
        return self._radii * self._scale

    def unstack_coordinates(self):
        """Return a list of all coordinates in the group."""
        return self._coordinates

    def unstack_positions(self):
        """Return a list of all coordinates in the group."""
        return self._positions

    def update_positions(self, axes, offset, scale):
        """Update element positions, give a axes basis and centre offset."""
        self._positions = np.dot(self._coordinates, axes) - offset
        self._axes = axes
        self._offset = offset
        self._scale = scale

    def get_max_zposition(self):
        """Return the maximum z-coordinate."""
        return self._positions[:, 2] + self.scaled_radii


class DrawElementsLine(DrawElementsBase):
    """Store a set of 3D-visualisation line elements."""

    etype = "line"

    def __init__(
        self, name, coordinates, element_properties=None, group_properties=None
    ):
        """Initialise the element group."""
        coordinates = np.array(coordinates)
        if coordinates.shape == (0,):
            coordinates = np.empty((0, 2, 3))
        super().__init__(name, coordinates, element_properties, group_properties)

    def unstack_coordinates(self):
        """Return a list of all coordinates in the group."""
        return np.concatenate((self._coordinates[:, 0, :], self._coordinates[:, 1, :]))

    def unstack_positions(self):
        """Return a list of all coordinates in the group."""
        return np.concatenate((self._positions[:, 0, :], self._positions[:, 1, :]))

    def update_positions(self, axes, offset, scale):
        """Update element positions, give a axes basis and centre offset."""
        self._positions = np.einsum("ijk, km -> ijm", self._coordinates, axes) - offset
        self._axes = axes
        self._offset = offset
        self._scale = scale

    def get_max_zposition(self):
        """Return the maximum z-coordinate."""
        return self._positions.max(axis=1)[:, 2]


class DrawElementsPoly(DrawElementsBase):
    """Store a set of 3D-visualisation polygon elements."""

    etype = "poly"

    def unstack_coordinates(self):
        """Return a list of all coordinates in the group."""
        planes = [np.array(plane) for plane in self._coordinates]
        if not planes:
            return np.empty((0, 3))
        return np.concatenate(planes)

    def unstack_positions(self):
        """Return a list of all coordinates in the group."""
        planes = [np.array(plane) for plane in self._positions]
        if not planes:
            return np.empty((0, 3))
        return np.concatenate(planes)

    def update_positions(self, axes, offset, scale):
        """Update element positions, give a axes basis and centre offset."""
        # TODO ideally would apply transform to all planes at once
        self._positions = [np.dot(plane, axes) - offset for plane in self._coordinates]
        self._axes = axes
        self._offset = offset
        self._scale = scale

    def get_max_zposition(self):
        """Return the maximum z-coordinate."""
        return np.array([plane[:, 2].max() for plane in self._positions])


class DrawGroup(Mapping):
    """Store and manipulate 3-D visualisation element groups."""

    def __init__(self, elements: List[DrawElementsBase]):
        """Store and manipulate 3-D visualisation element groups."""
        self._elements = OrderedDict([(el.name, el) for el in elements])

    def __getitem__(self, key):
        """Return an element group by name."""
        return self._elements[key]

    def __iter__(self):
        """Iterate over the element group names."""
        for key in self._elements:
            yield key

    def __len__(self):
        """Return the number of element groups."""
        return len(self._elements)

    def get_all_coordinates(self):
        """Return a list of all coordinates."""
        coordinates = [el.unstack_coordinates() for el in self._elements.values()]
        return np.concatenate(coordinates)

    def get_all_positions(self):
        """Return a list of all coordinates."""
        positions = [el.unstack_positions() for el in self._elements.values()]
        return np.concatenate(positions)

    def update_positions(self, axes=None, offset=None, scale=1):
        """Update element positions, give a axes basis and centre offset."""
        if axes is None:
            axes = np.identity(3)
        if offset is None:
            offset = np.zeros(3)
        for element in self._elements.values():
            element.update_positions(axes, offset, scale)

    def get_position_range(self):
        """Return the (minimum, maximum) coordinates."""
        min_positions = []
        max_positions = []
        for element in self._elements.values():
            positions = element.unstack_positions()
            if isinstance(element, DrawElementsSphere):  # type: DrawElementsSphere
                # TODO make more general
                min_positions.append(positions - element.scaled_radii[:, None])
                max_positions.append(positions + element.scaled_radii[:, None])
            else:
                min_positions.append(positions)
                max_positions.append(positions)
        return (
            np.concatenate(min_positions).min(0),
            np.concatenate(max_positions).max(0),
        )

    def yield_zorder(self):
        """Yield elements, in order of the z-coordinate."""
        keys = [(el.name, i) for el in self._elements.values() for i in range(len(el))]
        for i in np.concatenate(
            [el.get_max_zposition() for el in self._elements.values()]
        ).argsort():
            yield self[keys[i][0]][keys[i][1]]


def compute_element_coordinates(
    atoms,
    atom_radii,
    show_uc=True,
    uc_segments=None,
    show_bonds=False,
    bond_supercell=(1, 1, 1),
    miller_planes=None,
):
    """Compute (untransformed) coordinates, for elements in the visualisation.

    Parameters
    ----------
    atoms : ase.Atoms
    atom_radii : list or None
        mapping of atom index to atomic radii
    show_uc : bool
        show the unit cell
    uc_segments : float or None
        split unit cell lines into maximum length segments (improves z-order)
    show_bonds : bool
        show the atomic bonds
    bond_supercell : tuple
        the supercell of unit cell used for computing bonds
    miller_planes: list[tuple] or None
        list of (h, k, l, colour, thickness, as_poly) to project onto the unit cell,
        e.g. [(1, 0, 0, "blue", 1, True), (2, 2, 2, "green", 3.5, False)].

    Returns
    -------
    elements: dict
    all_coordinates: numpy.array

    """
    if show_uc:
        cvec_starts, cvec_ends = get_cell_coordinates(
            atoms.cell,
            show_repeats=atoms.info.get("unit_cell_repeat", None),
            segment_length=uc_segments,
        )
    else:
        cvec_starts = cvec_ends = np.zeros((0, 3))

    el_cell_lines = {"coordinates": np.stack((cvec_starts, cvec_ends), axis=1)}

    el_miller_lines = {"starts": [], "ends": [], "index": []}
    el_miller_planes = {"coordinates": [], "index": []}

    if miller_planes is not None:

        for i, (h, k, l, color, thickness, plane) in enumerate(miller_planes):
            miller_points = get_miller_coordinates(atoms.cell, (h, k, l)).tolist()
            if plane:
                el_miller_planes["coordinates"].append(miller_points)
                el_miller_planes["index"].append(i)
            else:
                el_miller_lines["starts"].extend(miller_points)
                el_miller_lines["ends"].extend(miller_points[1:] + [miller_points[0]])
                el_miller_lines["index"].extend([i for _ in miller_points])

    el_miller_lines["coordinates"] = np.stack(
        (
            el_miller_lines.pop("starts") or np.zeros((0, 3)),
            el_miller_lines.pop("ends") or np.zeros((0, 3)),
        ),
        axis=1,
    )
    el_miller_planes["coordinates"] = np.array(
        el_miller_planes["coordinates"] or np.zeros((0, 4, 3)), dtype=float
    )

    if show_bonds:
        atomscopy = atoms.copy()
        atomscopy.cell *= np.array(bond_supercell)[:, np.newaxis]
        bonds = compute_bonds(atomscopy, atom_radii)
        bond_atom_indices = [(bond[0], bond[1]) for bond in bonds]
    else:
        bonds = np.empty((0, 5), int)
        bond_atom_indices = []

    if len(bonds) > 0:
        positions = atoms.positions
        cell = np.array(bond_supercell)[:, np.newaxis] * atoms.cell
        a = positions[bonds[:, 0]]
        b = positions[bonds[:, 1]] + np.dot(bonds[:, 2:], cell) - a
        d = (b ** 2).sum(1) ** 0.5
        r = 0.65 * atom_radii
        x0 = (r[bonds[:, 0]] / d).reshape((-1, 1))
        x1 = (r[bonds[:, 1]] / d).reshape((-1, 1))
        bond_starts = a + b * x0
        b *= 1.0 - x0 - x1
        b[bonds[:, 2:].any(1)] *= 0.5
        bond_ends = bond_starts + b
    else:
        bond_starts = bond_ends = np.empty((0, 3))

    el_bond_lines = {
        "coordinates": np.stack((bond_starts, bond_ends), axis=1),
        "atom_index": bond_atom_indices,
    }

    return DrawGroup(
        [
            DrawElementsSphere("atoms", atoms.positions[:], atom_radii),
            DrawElementsLine("cell_lines", el_cell_lines["coordinates"]),
            DrawElementsLine(
                "bond_lines",
                el_bond_lines["coordinates"],
                element_properties={"atom_index": el_bond_lines["atom_index"]},
            ),
            DrawElementsLine(
                "miller_lines",
                el_miller_lines["coordinates"],
                element_properties={"index": el_miller_lines["index"]},
            ),
            DrawElementsPoly(
                "miller_planes",
                el_miller_planes["coordinates"],
                element_properties={"index": el_miller_planes["index"]},
            ),
        ]
    )
