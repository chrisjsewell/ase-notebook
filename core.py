"""Implementation agnostics visualisation functions."""
from collections import namedtuple, OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from itertools import product
from math import ceil, cos, radians, sin
from typing import List

import numpy as np


def compute_projection(element_group, wsize, rotation, whitespace=1.3):
    """Compute the center and scale of the projection."""
    element_group.update_positions(rotation)
    min_coord, max_coord = element_group.get_position_range()
    center = np.dot(rotation, (min_coord + max_coord) / 2)
    s = whitespace * (max_coord - min_coord)
    width, height = wsize
    if s[0] * height < s[1] * width:
        scale = height / s[1]
    elif s[0] > 0.0001:
        scale = width / s[0]
    else:
        scale = 1.0
    return center, scale


def get_rotation_matrix(rotations, init_rotation=None):
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
    cell, origin=(0.0, 0.0, 0.0), show_repeats=None, dash_pattern=None
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

    if dash_pattern:
        # split lines into a dash pattern
        dlength, dgap = dash_pattern
        new_lines = []
        for (start, end) in lines:
            new_start = start
            total_length = np.linalg.norm(end - start)
            dash_fraction = (dlength + dgap) / total_length
            length_fraction = dlength / total_length
            ndashes = int(ceil(total_length / (dlength + dgap)))
            for n in range(ndashes - 1):
                dash_end = start + (end - start) * (
                    (dash_fraction * n) + length_fraction
                )
                new_lines.append([new_start, dash_end])
                new_start = start + (end - start) * dash_fraction * (n + 1)
            # TODO remove last gap fraction (if present)
            # or, better, start with a fraction of dlength, so start/end are symmetric
            new_lines.append([new_start, end])

        lines = np.array(new_lines, dtype=float)

    return lines[:, 0], lines[:, 1]


def get_miller_coordinates(cell, miller):
    """Compute the points at which a miller index intercepts with a unit cell boundary."""
    vec_a, vec_b, vec_c = np.array(cell, dtype=float)
    h_val, k_val, l_val = miller

    if h_val < 0 or k_val < 0 or l_val < 0:
        # TODO compute negative miller intercepts
        # look at script in https://www.doitpoms.ac.uk/tlplib/miller_indices/printall.php
        # they appear to use a transpose
        raise NotImplementedError("h, k or l less than zero")

    h_is_zero, k_is_zero, l_is_zero = np.isclose(miller, 0)

    mod_a = np.inf if h_is_zero else vec_a / h_val
    mod_b = np.inf if k_is_zero else vec_b / k_val
    mod_c = np.inf if l_is_zero else vec_c / l_val

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
    pbc_bonds = bonds[:n2][i]
    bonds[n2:, 0] = pbc_bonds[:, 1]
    bonds[n2:, 1] = pbc_bonds[:, 0]
    bonds[n2:, 2:] = -pbc_bonds[:, 2:]
    return bonds


class Element(object):
    """Frozen dictionary with attribute access."""

    def __init__(self, **kwargs):
        """Initialise element."""
        self._kwargs = kwargs

    def __dir__(self):
        """Get the attributes."""
        return list(self._kwargs.keys()) + ["get"]

    def get(self, key, default):
        """Return key or default."""
        if key in self:
            return self[key]
        else:
            return default

    def __repr__(self):
        """Represent object."""
        sig = ", ".join([f"{k}={self._kwargs[k]}" for k in sorted(self._kwargs)])
        return f"Element({sig})"

    def __getitem__(self, key):
        """Return key."""
        return self._kwargs[key]

    def __getattr__(self, key):
        """Return key."""
        return self._kwargs[key]

    def __setattr__(self, name, key):
        """Return key."""
        if name != "_kwargs":
            raise AttributeError("Element attributes are frozen")
        return super().__setattr__(name, key)

    def __contains__(self, key):
        """Test if key in object."""
        return key in self._kwargs


class DrawElementsBase:
    """Abstract base class to store a set of 3D-visualisation elements."""

    etype = None
    _protected_keys = ("name", "type", "position", "get")

    def __init__(
        self, name, coordinates, element_properties=None, group_properties=None
    ):
        """Initialise the element group."""
        self.name = name
        self._coordinates = coordinates
        self._positions = coordinates
        self._axes = np.identity(3)
        self._offset = np.zeros(3)
        self._el_props = {}
        self._grp_props = {}

        for key, val in (element_properties or {}).items():
            self.set_property(key, val, element=True)
        for key, val in (group_properties or {}).items():
            self.set_property(key, val, element=False)

    @property
    def element_properties(self):
        """Return per element properties."""
        output = deepcopy(self._el_props)
        output["positions"] = np.array(self._positions)
        return output

    @property
    def group_properties(self):
        """Return element group properties."""
        return deepcopy(self._grp_props)

    def set_property(self, name, value, element=False):
        """Set a group or per element property."""
        if name in self._protected_keys:
            raise KeyError(f"{name} is a protected key name")
        if element:
            assert len(value) == len(self._coordinates)
            assert (
                name not in self._grp_props
            ), f"{name} is already set as a group property"
            self._el_props[name] = value
        else:
            assert (
                name not in self._el_props
            ), f"{name} is already set as an element property"
            self._grp_props[name] = value

    def set_property_many(self, properties, element=False):
        """Set multiple group or per element properties."""
        for key, val in properties.items():
            self.set_property(key, val, element=element)

    def get_elements_property(self, name):
        """Return a single property."""
        if name == "position":
            return np.array(self._positions)
        if name in self._el_props:
            return [i for i in self._el_props[name]]
        if name in self._grp_props:
            return [self._grp_props[name] for _ in range(len(self))]
        raise KeyError(f"{name} not in properties")

    def __len__(self):
        """Return the number of elements."""
        return len(self._coordinates)

    def __getitem__(self, index):
        """Return a single element."""
        try:
            index = int(index)
        except ValueError:
            raise TypeError(f"index must be an integer: {index}")
        return Element(
            name=self.name,
            type=self.etype,
            position=self._positions[index],
            **dict(
                [(str(k), v[index]) for k, v in self._el_props.items()]
                + [(str(k), v) for k, v in self._grp_props.items()]
            ),
        )

    def __iter__(self):
        """Iterate over elements."""
        for i in range(len(self)):
            yield self[i]

    def unstack_coordinates(self):
        """Return a list of all coordinates in the group."""
        raise NotImplementedError

    def unstack_positions(self):
        """Return a list of all coordinates in the group."""
        raise NotImplementedError

    def update_positions(self, axes, offset, **kwargs):
        """Update element positions, give a axes basis and centre offset."""
        raise NotImplementedError

    def get_max_zposition(self):
        """Return the maximum z-coordinate."""
        raise NotImplementedError


class DrawElementsSphere(DrawElementsBase):
    """Store a set of 3D-visualisation sphere elements."""

    etype = "sphere"
    _protected_keys = ("name", "type", "position", "sradius", "get")

    def __init__(
        self,
        name,
        coordinates,
        radii,
        element_properties=None,
        group_properties=None,
        radii_scale=1.0,
    ):
        """Initialise the element group."""
        coordinates = np.array(coordinates)
        if coordinates.shape == (0,):
            coordinates = np.empty((0, 3))
        super().__init__(name, coordinates, element_properties, group_properties)
        self._radii = np.array(radii)
        self._radii_scale = radii_scale

    def __getitem__(self, index):
        """Return a single element."""
        try:
            index = int(index)
        except ValueError:
            raise TypeError(f"index must be an integer: {index}")
        return Element(
            name=self.name,
            type=self.etype,
            position=self._positions[index],
            sradius=self.scaled_radii[index],
            **dict(
                [(str(k), v[index]) for k, v in self._el_props.items()]
                + [(str(k), v) for k, v in self._grp_props.items()]
            ),
        )

    @property
    def scaled_radii(self):
        """Return the scaled radii, for each sphere."""
        return self._radii * self._radii_scale

    def unstack_coordinates(self):
        """Return a list of all coordinates in the group."""
        return self._coordinates

    def unstack_positions(self):
        """Return a list of all coordinates in the group."""
        return self._positions

    def update_positions(self, axes, offset, radii_scale, **kwargs):
        """Update element positions, give a axes basis and centre offset."""
        self._positions = np.dot(self._coordinates, axes) - offset
        self._axes = axes
        self._offset = offset
        self._radii_scale = radii_scale

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

    def update_positions(self, axes, offset, **kwargs):
        """Update element positions, give a axes basis and centre offset."""
        self._positions = np.einsum("ijk, km -> ijm", self._coordinates, axes) - offset
        self._axes = axes
        self._offset = offset

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

    def update_positions(self, axes, offset, **kwargs):
        """Update element positions, give a axes basis and centre offset."""
        # TODO ideally would apply transform to all planes at once
        self._positions = [np.dot(plane, axes) - offset for plane in self._coordinates]
        self._axes = axes
        self._offset = offset

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

    def update_positions(self, axes=None, offset=None, radii_scale=1):
        """Update element positions, give a axes basis and centre offset."""
        if axes is None:
            axes = np.identity(3)
        if offset is None:
            offset = np.zeros(3)
        for element in self._elements.values():
            element.update_positions(axes, offset, radii_scale=radii_scale)

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
        keys = [(el.name, i) for el in self.values() for i in range(len(el))]
        z_positions = np.concatenate([el.get_max_zposition() for el in self.values()])
        for i in z_positions.argsort():
            yield i, self[keys[i][0]][keys[i][1]]


def initialise_element_groups(
    atoms,
    atom_radii,
    show_unit_cell=True,
    uc_dash_pattern=None,
    show_bonds=False,
    bond_supercell=(1, 1, 1),
    miller_planes=None,
    miller_planes_as_lines=False,
):
    """Compute (untransformed) coordinates, for elements in the visualisation.

    Parameters
    ----------
    atoms : ase.Atoms
    atom_radii : list or None
        mapping of atom index to atomic radii
    show_unit_cell : bool
        show the unit cell
    uc_dash_pattern : tuple or None
        split unit cell lines into dash pattern (line_length, gap_length)
    show_bonds : bool
        show the atomic bonds
    bond_supercell : tuple
        the supercell of unit cell used for computing bonds
    miller_planes: list[dict] or None
        list of miller planes to project onto the unit cell
    miller_planes_as_lines: bool
        whether to create miller planes as a group of lines or a solid plane

    Returns
    -------
    elements: dict
    all_coordinates: numpy.array

    """
    if show_unit_cell:
        cvec_starts, cvec_ends = get_cell_coordinates(
            atoms.cell,
            show_repeats=atoms.info.get("unit_cell_repeat", None),
            dash_pattern=uc_dash_pattern,
        )
    else:
        cvec_starts = cvec_ends = np.zeros((0, 3))

    el_cell_lines = {"coordinates": np.stack((cvec_starts, cvec_ends), axis=1)}

    el_miller_lines = {"starts": [], "ends": [], "index": []}
    el_miller_planes = {"coordinates": [], "index": []}

    if miller_planes is not None:

        for i, plane in enumerate(miller_planes):
            miller_points = get_miller_coordinates(
                atoms.cell, [plane[n] for n in "hkl"]
            ).tolist()
            if miller_planes_as_lines:
                el_miller_lines["starts"].extend(miller_points)
                el_miller_lines["ends"].extend(miller_points[1:] + [miller_points[0]])
                el_miller_lines["index"].extend([i for _ in miller_points])
            else:
                el_miller_planes["coordinates"].append(miller_points)
                el_miller_planes["index"].append(i)

    el_miller_lines["coordinates"] = np.stack(
        (
            el_miller_lines.pop("starts") or np.zeros((0, 3)),
            el_miller_lines.pop("ends") or np.zeros((0, 3)),
        ),
        axis=1,
    )
    # el_miller_planes["coordinates"] = np.array(
    #     el_miller_planes["coordinates"] or np.zeros((0, 4, 3)), dtype=float
    # )

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


VestaElement = namedtuple(
    "VestaElement", ["element", "radius", "r2", "r3", "r", "g", "b"]
)

VESTA_ELEMENT_INFO = (
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("H", 0.46, 1.2, 0.2, 1.0, 0.8, 0.8),
    ("He", 1.22, 1.4, 1.22, 0.98907, 0.91312, 0.81091),
    ("Li", 1.57, 1.4, 0.59, 0.52731, 0.87953, 0.4567),
    ("Be", 1.12, 1.4, 0.27, 0.37147, 0.8459, 0.48292),
    ("B", 0.81, 1.4, 0.11, 0.1249, 0.63612, 0.05948),
    ("C", 0.77, 1.7, 0.15, 0.5043, 0.28659, 0.16236),
    ("N", 0.74, 1.55, 1.46, 0.69139, 0.72934, 0.9028),
    ("O", 0.74, 1.52, 1.4, 0.99997, 0.01328, 0.0),
    ("F", 0.72, 1.47, 1.33, 0.69139, 0.72934, 0.9028),
    ("Ne", 1.6, 1.54, 1.6, 0.99954, 0.21788, 0.71035),
    ("Na", 1.91, 1.54, 1.02, 0.97955, 0.86618, 0.23787),
    ("Mg", 1.6, 1.54, 0.72, 0.98773, 0.48452, 0.0847),
    ("Al", 1.43, 1.54, 0.39, 0.50718, 0.70056, 0.84062),
    ("Si", 1.18, 2.1, 0.26, 0.10596, 0.23226, 0.98096),
    ("P", 1.1, 1.8, 0.17, 0.75557, 0.61256, 0.76425),
    ("S", 1.04, 1.8, 1.84, 1.0, 0.98071, 0.0),
    ("Cl", 0.99, 1.75, 1.81, 0.19583, 0.98828, 0.01167),
    ("Ar", 1.92, 1.88, 1.92, 0.81349, 0.99731, 0.77075),
    ("K", 2.35, 1.88, 1.51, 0.63255, 0.13281, 0.96858),
    ("Ca", 1.97, 1.88, 1.12, 0.35642, 0.58863, 0.74498),
    ("Sc", 1.64, 1.88, 0.745, 0.71209, 0.3893, 0.67279),
    ("Ti", 1.47, 1.88, 0.605, 0.47237, 0.79393, 1.0),
    ("V", 1.35, 1.88, 0.58, 0.9, 0.1, 0.0),
    ("Cr", 1.29, 1.88, 0.615, 0.0, 0.0, 0.62),
    ("Mn", 1.37, 1.88, 0.83, 0.66148, 0.03412, 0.62036),
    ("Fe", 1.26, 1.88, 0.78, 0.71051, 0.44662, 0.00136),
    ("Co", 1.25, 1.88, 0.745, 0.0, 0.0, 0.68666),
    ("Ni", 1.25, 1.88, 0.69, 0.72032, 0.73631, 0.74339),
    ("Cu", 1.28, 1.88, 0.73, 0.1339, 0.28022, 0.86606),
    ("Zn", 1.37, 1.88, 0.74, 0.56123, 0.56445, 0.50799),
    ("Ga", 1.53, 1.88, 0.62, 0.62292, 0.89293, 0.45486),
    ("Ge", 1.22, 1.88, 0.53, 0.49557, 0.43499, 0.65193),
    ("As", 1.21, 1.85, 0.335, 0.45814, 0.81694, 0.34249),
    ("Se", 1.04, 1.9, 1.98, 0.6042, 0.93874, 0.06122),
    ("Br", 1.14, 1.85, 1.96, 0.49645, 0.19333, 0.01076),
    ("Kr", 1.98, 2.02, 1.98, 0.98102, 0.75805, 0.95413),
    ("Rb", 2.5, 2.02, 1.61, 1.0, 0.0, 0.6),
    ("Sr", 2.15, 2.02, 1.26, 0.0, 1.0, 0.15259),
    ("Y", 1.82, 2.02, 1.019, 0.40259, 0.59739, 0.55813),
    ("Zr", 1.6, 2.02, 0.72, 0.0, 1.0, 0.0),
    ("Nb", 1.47, 2.02, 0.64, 0.29992, 0.70007, 0.46459),
    ("Mo", 1.4, 2.02, 0.59, 0.70584, 0.52602, 0.68925),
    ("Tc", 1.35, 2.02, 0.56, 0.80574, 0.68699, 0.79478),
    ("Ru", 1.34, 2.02, 0.62, 0.81184, 0.72113, 0.68089),
    ("Rh", 1.34, 2.02, 0.665, 0.80748, 0.82205, 0.67068),
    ("Pd", 1.37, 2.02, 0.86, 0.75978, 0.76818, 0.72454),
    ("Ag", 1.44, 2.02, 1.15, 0.72032, 0.73631, 0.74339),
    ("Cd", 1.52, 2.02, 0.95, 0.95145, 0.12102, 0.86354),
    ("In", 1.67, 2.02, 0.8, 0.84378, 0.50401, 0.73483),
    ("Sn", 1.58, 2.02, 0.69, 0.60764, 0.56052, 0.72926),
    ("Sb", 1.41, 2.0, 0.76, 0.84627, 0.51498, 0.31315),
    ("Te", 1.37, 2.06, 2.21, 0.67958, 0.63586, 0.32038),
    ("I", 1.33, 1.98, 2.2, 0.55914, 0.122, 0.54453),
    ("Xe", 2.18, 2.16, 0.48, 0.60662, 0.63218, 0.97305),
    ("Cs", 2.72, 2.16, 1.74, 0.05872, 0.99922, 0.72578),
    ("Ba", 2.24, 2.16, 1.42, 0.11835, 0.93959, 0.17565),
    ("La", 1.88, 2.16, 1.16, 0.3534, 0.77057, 0.28737),
    ("Ce", 1.82, 2.16, 0.97, 0.82055, 0.99071, 0.02374),
    ("Pr", 1.82, 2.16, 1.126, 0.9913, 0.88559, 0.02315),
    ("Nd", 1.82, 2.16, 1.109, 0.98701, 0.5556, 0.02744),
    ("Pm", 1.81, 2.16, 1.093, 0.0, 0.0, 0.96),
    ("Sm", 1.81, 2.16, 1.27, 0.99042, 0.02403, 0.49195),
    ("Eu", 2.06, 2.16, 1.066, 0.98367, 0.03078, 0.83615),
    ("Gd", 1.79, 2.16, 1.053, 0.75325, 0.01445, 1.0),
    ("Tb", 1.77, 2.16, 1.04, 0.44315, 0.01663, 0.99782),
    ("Dy", 1.77, 2.16, 1.027, 0.1939, 0.02374, 0.99071),
    ("Ho", 1.76, 2.16, 1.015, 0.02837, 0.25876, 0.98608),
    ("Er", 1.75, 2.16, 1.004, 0.28688, 0.45071, 0.23043),
    ("Tm", 1.0, 2.16, 0.994, 0.0, 0.0, 0.88),
    ("Yb", 1.94, 2.16, 0.985, 0.15323, 0.99165, 0.95836),
    ("Lu", 1.72, 2.16, 0.977, 0.15097, 0.99391, 0.71032),
    ("Hf", 1.59, 2.16, 0.71, 0.70704, 0.70552, 0.3509),
    ("Ta", 1.47, 2.16, 0.64, 0.71952, 0.60694, 0.33841),
    ("W", 1.41, 2.16, 0.6, 0.55616, 0.54257, 0.50178),
    ("Re", 1.37, 2.16, 0.53, 0.70294, 0.69401, 0.55789),
    ("Os", 1.35, 2.16, 0.63, 0.78703, 0.69512, 0.47379),
    ("Ir", 1.36, 2.16, 0.625, 0.78975, 0.81033, 0.45049),
    ("Pt", 1.39, 2.16, 0.625, 0.79997, 0.77511, 0.75068),
    ("Au", 1.44, 2.16, 1.37, 0.99628, 0.70149, 0.22106),
    ("Hg", 1.55, 2.16, 1.02, 0.8294, 0.72125, 0.79823),
    ("Tl", 1.71, 2.16, 0.885, 0.58798, 0.53854, 0.42649),
    ("Pb", 1.75, 2.16, 1.19, 0.32386, 0.32592, 0.35729),
    ("Bi", 1.82, 2.16, 1.03, 0.82428, 0.18732, 0.97211),
    ("Po", 1.77, 2.16, 0.94, 0.0, 0.0, 1.0),
    ("At", 0.62, 2.16, 0.62, 0.0, 0.0, 1.0),
    ("Rn", 0.8, 2.16, 0.8, 1.0, 1.0, 0.0),
    ("Fr", 1.0, 2.16, 1.8, 0.0, 0.0, 0.0),
    ("Ra", 2.35, 2.16, 1.48, 0.42959, 0.66659, 0.34786),
    ("Ac", 2.03, 2.16, 1.12, 0.39344, 0.62101, 0.45034),
    ("Th", 1.8, 2.16, 1.05, 0.14893, 0.99596, 0.47106),
    ("Pa", 1.63, 2.16, 0.78, 0.16101, 0.98387, 0.20855),
    ("U", 1.56, 2.16, 0.73, 0.47774, 0.63362, 0.66714),
    ("Np", 1.56, 2.16, 0.75, 0.3, 0.3, 0.3),
    ("Pu", 1.64, 2.16, 0.86, 0.3, 0.3, 0.3),
    ("Am", 1.73, 2.16, 0.975, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
    ("XX", 0.8, 1.0, 0.8, 0.3, 0.3, 0.3),
)
