"""Implementation agnostics visualisation functions."""
from itertools import product
from math import ceil

import numpy as np


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


def compute_element_coordinates(
    atoms,
    show_uc=True,
    uc_segments=None,
    show_bonds=False,
    bond_supercell=(1, 1, 1),
    atom_radii=None,
    miller_planes=None,
):
    """Compute (untransformed) coordinates, for elements in the visualisation.

    Parameters
    ----------
    atoms : ase.Atoms
    show_uc : bool
        show the unit cell
    uc_segments : float or None
        split unit cell lines into maximum length segments (improves z-order)
    show_bonds : bool
        show the atomic bonds
    bond_supercell : tuple
        the supercell of unit cell used for computing bonds
    atom_radii : list or None
        mapping of atom index to atomic radii, used for computing bonds
    miller_planes: list[tuple] or None
        list of (h, k, l, colour, thickness, as_poly) to project onto the unit cell,
        e.g. [(1, 0, 0, "blue", 1, True), (2, 2, 2, "green", 3.5, False)].

    Returns
    -------
    elements: dict
    all_coordinates: numpy.array

    """
    if show_bonds and atom_radii is None:
        raise ValueError("must supply atom_radii, if computing bonds")

    el_atoms = {"coordinates": atoms.positions, "type": "point"}

    if show_uc:
        cvec_starts, cvec_ends = get_cell_coordinates(
            atoms.cell,
            show_repeats=atoms.info.get("unit_cell_repeat", None),
            segment_length=uc_segments,
        )
    else:
        cvec_starts = cvec_ends = np.zeros((0, 3))

    el_cell_lines = {
        "coordinates": np.stack((cvec_starts, cvec_ends), axis=1),
        "type": "line",
    }

    el_miller_lines = {"starts": [], "ends": [], "index": [], "type": "line"}
    el_miller_planes = {"coordinates": [], "index": [], "type": "plane"}
    all_miller_points = []

    if miller_planes is not None:

        for i, (h, k, l, color, thickness, plane) in enumerate(miller_planes):
            miller_points = get_miller_coordinates(atoms.cell, (h, k, l)).tolist()
            all_miller_points.extend(miller_points)
            if plane:
                el_miller_planes["coordinates"].append(
                    miller_points
                    if len(miller_points) == 4
                    else miller_points + [[np.nan, np.nan, np.nan]]
                )
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
        "type": "line",
        "atom_index": bond_atom_indices,
    }

    all_coordinates = np.concatenate(
        (
            atoms.positions[:],
            cvec_starts,
            cvec_ends,
            all_miller_points or np.empty((0, 3)),
            bond_starts,
            bond_ends,
        )
    )

    return (
        {
            "atoms": el_atoms,
            "cell_lines": el_cell_lines,
            "bond_lines": el_bond_lines,
            "miller_lines": el_miller_lines,
            "miller_planes": el_miller_planes,
        },
        all_coordinates,
    )


if __name__ == "__main__":
    _self = None
    atoms = None
    compute_element_coordinates(
        atoms,
        show_uc=_self.showing_cell(),
        uc_segments=_self.config["unit_cell_segmentation"],
        show_bonds=_self.showing_bonds(),
        bond_supercell=_self.images.repeat,
        element_colors=_self.colors,
        atom_radii=_self.get_covalent_radii(),
        miller_planes=_self.get_millers() if _self.showing_millers() else None,
    )
