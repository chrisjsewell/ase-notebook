"""Implementation agnostics visualisation functions."""
from itertools import product
from math import ceil, cos, radians, sin

import numpy as np

import ase_notebook.draw_elements as draw


def triangle_normal(a, b, c):
    """Compute the normal of three points."""
    a, b, c = [np.array(i) for i in (a, b, c)]
    return np.cross(b - a, c - a).tolist()


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

    return draw.DrawGroup(
        [
            draw.DrawElementsSphere("atoms", atoms.positions[:], atom_radii),
            draw.DrawElementsLine("cell_lines", el_cell_lines["coordinates"]),
            draw.DrawElementsLine(
                "bond_lines",
                el_bond_lines["coordinates"],
                element_properties={"atom_index": el_bond_lines["atom_index"]},
            ),
            draw.DrawElementsLine(
                "miller_lines",
                el_miller_lines["coordinates"],
                element_properties={"index": el_miller_lines["index"]},
            ),
            draw.DrawElementsPoly(
                "miller_planes",
                el_miller_planes["coordinates"],
                element_properties={"index": el_miller_planes["index"]},
            ),
        ]
    )
