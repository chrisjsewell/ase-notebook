"""Module for creating a information string of atomic sites."""
from math import acos, pi, sqrt

from ase.data import atomic_names
from ase.data import chemical_symbols
from ase.gui.i18n import _
from ase.gui.utils import get_magmoms
import numpy as np


def create_formula(atomic_numbers):
    """Create a formula from a list of atomic numbers."""
    an_count = {}
    for z in atomic_numbers:
        an_count.setdefault(z, 1)
        an_count[z] += 1

    strings = []
    for z in sorted(an_count.keys()):
        strings.append(
            chemical_symbols[z] + ("" if an_count[z] == 1 else str(an_count[z]))
        )
    return "".join(strings)


def create_info_lines(atoms, indices, ordered_indices=None):
    """Create a string of information about the selected atom(s).

    Return
    ------
    list[str]

    """
    ordered_indices = ordered_indices or indices
    num_selected = len(indices)

    if num_selected == 0:
        return []

    a_nums = atoms.numbers[indices]
    symbols = [chemical_symbols[z] for z in a_nums]
    positions = atoms.positions[indices]

    if num_selected == 1:
        x, y, z = positions[0]
        name = atomic_names[a_nums[0]]
        text = [
            f"#{indices[0]} {name} ({symbols[0]})",
            f"({x:.3f}, {y:.3f}, {z:.3f}) Å",
            _(f"tag={atoms.get_tags()[indices[0]]}"),
        ]
        magmoms = get_magmoms(atoms)
        if magmoms.any():
            text.append(_(f"mom={magmoms[indices][0]:1.2f}"))
        charges = atoms.get_initial_charges()
        if charges.any():
            text.append(_(f"q={charges[indices][0]:1.2f}"))
        known_arrays = [
            "numbers",
            "positions",
            "forces",
            "momenta",
            "initial_charges",
            "initial_magmoms",
        ]
        for key in atoms.arrays:
            if key not in known_arrays:
                val = atoms.get_array(key)[indices[0]]
                if val is not None:
                    try:
                        text.append("{0}={1:g}".format(key, val))
                    except ValueError:
                        text.append("{0}={1}".format(key, val))

        return text

    if num_selected == 2:
        dist = np.linalg.norm(positions[0] - positions[1])
        return [f"Bond Distance {symbols[0]}-{symbols[0]}: {dist:.3f} Å"]

    if num_selected == 3:
        distances = []
        for c in range(3):
            vector = positions[c] - positions[(c + 1) % 3]
            distances.append(np.dot(vector, vector))
        angles = []
        for c in range(3):
            t1 = 0.5 * (distances[c] + distances[(c + 1) % 3] - distances[(c + 2) % 3])
            t2 = sqrt(distances[c] * distances[(c + 1) % 3])
            try:
                t3 = acos(t1 / t2)
            except ValueError:
                if t1 > 0:
                    t3 = 0
                else:
                    t3 = pi
            angles.append(t3 * 180 / pi)
        return ["Valence Angle %s-%s-%s: %.1f°, %.1f°, %.1f°" % tuple(symbols + angles)]

    if len(ordered_indices) == 4:
        angle = atoms.get_dihedral(*ordered_indices, mic=True)
        return [
            "%s %s → %s → %s → %s: %.1f°" % tuple([_("Dihedral")] + symbols + [angle])
        ]

    return [f"Formula: {create_formula(a_nums)}"]
