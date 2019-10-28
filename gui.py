"""A module for visualising structures.

The module subclasses ase (v3.18.0) classes, to add additional functionality.
"""
from time import time

from ase.data import atomic_numbers
from ase.data import covalent_radii as default_covalent_radii
from ase.gui import ui
from ase.gui.gui import GUI
from ase.gui.images import Images
from ase.gui.status import Status
from ase.gui.view import GREEN, PURPLE, View
import numpy as np

from aiida_2d.visualize.color import Color
from aiida_2d.visualize.core import initialise_element_groups


def get_default_settings(overrides=None):
    """Return the default setting for the GUI."""
    # values commented out are never actually utilised in the code
    dct = {
        # "gui_graphs_string": "i, e - E[-1]",  # default for the graph command
        "gui_foreground_color": "#000000",
        "gui_background_color": "#ffffff",
        # "covalent_radii": None,
        "radii_scale": 0.89,
        "force_vector_scale": 1.0,
        "velocity_vector_scale": 1.0,
        "show_unit_cell": True,
        "unit_cell_segmentation": None,
        "show_axes": True,
        "show_bonds": False,
        "show_millers": True,
        # "shift_cell": False,
        "swap_mouse": False,
    }
    if overrides:
        dct.update(overrides)
    return dct


def get_ghost_settings(overrides=None):
    """Return the default setting for ghost atoms."""
    dct = {
        "display": True,
        "cross": False,
        "label": False,
        "lighten": 0.0,
        "opacity": 0.4,
        "linewidth": 0,
    }
    if overrides:
        dct.update(overrides)
    return dct


class AtomImages(Images):
    """A subclass of the ase ``Images``, but with additional functionality, for setting radii."""

    def __init__(self, atoms_list, element_radii=None, radii_scale=0.89):
        """Initialise the atom images.

        Parameters
        ----------
        atoms_list : list[ase.Atoms]
        element_radii : list[float]
            radii for each atomic number (default to ase covalent)
        radii_scale : float
            scale all atomic_radii

        """
        if element_radii:
            self.covalent_radii = np.array(element_radii, dtype=float)
        else:
            self.covalent_radii = default_covalent_radii.copy()
        # In the base class, self.config is set, but it is only used for radii scale
        # self.config = get_default_settings()
        # self.atom_scale = self.config["radii_scale"]
        self.atom_scale = radii_scale
        self.initialize(atoms_list)


class AtomGui(GUI):
    """A subclass of the ase ``GUI``, but with additional functionality."""

    def __init__(
        self,
        images=None,
        config_settings=None,
        rotations="",
        element_colors=None,
        label_sites=0,
        ghost_settings=None,
        miller_planes=(),
    ):
        """Initialise the GUI.

        Parameters
        ----------
        images : ase.gui.images.Images
            list of ase.Atoms, with some settings for visualisations (mainly radii)
        config_settings: dict or None
            configuration settings to overrid defaults (e.g. {"show_axes": False})
        rotations : str
            string format of unit cell rotations '50x,-10y,120z' (note: order matters)
        element_colors: list[tuple]
            hex colour for each atomic number (defaults to 'jmol' scheme)
        label_sites: int
            0=None, 1=index, 2=magmoms, 3=symbols, 4=charges (see ``View.update_labels``)
        ghost_settings: dict or None
            How to display atoms labelled as ghosts (overrides for default settings).
            Ghost atoms are determined if atoms.arrays["ghost"] is set
        miller_planes: list[tuple]
            list of (h, k, l, colour, thickness, as_poly) to project onto the unit cell,
            e.g. [(1, 0, 0, "blue", 1, True), (2, 2, 2, "green", 3.5, False)].

        """
        if not isinstance(images, Images):
            images = Images(images)

        self.images = images
        self.observers = []

        self.config = get_default_settings(config_settings)
        self.ghost_settings = get_ghost_settings(ghost_settings)
        self._miller_planes = miller_planes

        menu = self.get_menu_data()

        self.window = ui.ASEGUIWindow(
            close=self.exit,
            menu=menu,
            config=self.config,
            scroll=self.scroll,
            scroll_event=self.scroll_event,
            press=self.press,
            move=self.move,
            release=self.release,
            resize=self.resize,
        )
        self.window["show-labels"] = label_sites

        View.__init__(self, rotations)
        if element_colors:
            self.colors = dict(enumerate(element_colors))
        Status.__init__(self)

        self.subprocesses = []  # list of external processes
        self.movie_window = None
        self.vulnerable_windows = []
        self.simulation = {}  # Used by modules on Calculate menu.
        self.module_state = {}  # Used by modules to store their state.

        self.arrowkey_mode = self.ARROWKEY_SCAN
        self.move_atoms_mask = None

        self.set_frame(len(self.images) - 1, focus=True)

        # Used to move the structure with the mouse
        self.prev_pos = None
        self.last_scroll_time = self.t0 = time()
        self.orig_scale = self.scale

        self.xy = None

        if len(self.images) > 1:
            self.movie()

    def release(self, event):
        """Handle release event."""
        # fix an error raised in GUI class
        if self.xy is None:
            self.xy = (event.x, event.y)

        super().release(event)

    def showing_millers(self):
        """Return whether to display planes."""
        return self.config["show_millers"]

    def get_millers(self):
        """Return list of miller indices to project onto the unit cell, e.g. [(h, k, l), ...]."""
        return self._miller_planes[:]

    def get_miller_color(self, index):
        """Return colour of miller plane."""
        return self._miller_planes[index].get("color", "blue")

    def get_miller_thickness(self, index):
        """Return thickness of miller plane lines."""
        return self._miller_planes[index].get("stroke_width", "blue")

    def set_atoms(self, atoms):
        """Set the atoms, unit cell(s) and bonds to draw.

        This is overridden from ``View``, in order to

        - set bond colors, specific to the atom types at each end of the bond.
        - use a modified ``get_cell_coordinates`` function,
          which returns cartesian coordinates, rather than fractional
          (since they were just converted to cartesian anyway)
          and can create multiple cells (for each repeat)
        - compute miller index planes, by points that intercept with the unit cell

        """
        elements = initialise_element_groups(
            atoms,
            atom_radii=self.get_covalent_radii(),
            show_uc=self.showing_cell(),
            uc_segments=self.config["unit_cell_segmentation"],
            show_bonds=self.showing_bonds(),
            bond_supercell=self.images.repeat,
            miller_planes=self.get_millers() if self.showing_millers() else None,
        )
        self.elements = elements

        # record all positions (atoms first) with legacy array name, for use by View.focus
        self.X = elements.get_all_coordinates()
        # record atom positions with legacy array name, used by View.move
        self.X_pos = self.elements["atoms"]._positions.copy()

    def circle(self, color, selected, *bbox, tags=()):
        """Add a circle element.

        Replacement to ``View.circle``, but with added tags option

        """
        if selected:
            outline = "#004500"
            width = 3
        else:
            outline = "black"
            width = 1
        self.window.canvas.create_oval(
            *tuple(int(x) for x in bbox),
            fill=color,
            outline=outline,
            width=width,
            tags=tags,
        )

    def _draw_atom(
        self, obj_indx, diameter, atom_lbound, selected, color, ghost, tag, movecolor
    ):
        """Draw a single atom."""
        circle = self.circle
        arc = self.window.arc

        if ghost:
            atom_color = Color(color)
            atom_color.luminance += (1 - atom_color.luminance) * self.ghost_settings[
                "lighten"
            ]
            atom_color = atom_color.web
        else:
            atom_color = color

        if "occupancy" in self.atoms.info:
            site_occ = self.atoms.info["occupancy"][tag]
            # first an empty circle if a site is not fully occupied
            if (np.sum([vector for vector in site_occ.values()])) < 1.0:
                fill = "#ffffff"
                circle(
                    fill,
                    selected,
                    atom_lbound[0],
                    atom_lbound[1],
                    atom_lbound[0] + diameter,
                    atom_lbound[1] + diameter,
                )
            start = 0
            # start with the dominant species
            for sym, occ in sorted(site_occ.items(), key=lambda x: x[1], reverse=True):
                if np.round(occ, decimals=4) == 1.0:
                    circle(
                        atom_color,
                        selected,
                        atom_lbound[0],
                        atom_lbound[1],
                        atom_lbound[0] + diameter,
                        atom_lbound[1] + diameter,
                        tags=("atom-circle", "atom-ghost")
                        if ghost
                        else ("atom-circle",),
                    )
                else:
                    # TODO alter for ghost
                    extent = 360.0 * occ
                    arc(
                        self.colors[atomic_numbers[sym]],
                        selected,
                        start,
                        extent,
                        atom_lbound[0],
                        atom_lbound[1],
                        atom_lbound[0] + diameter,
                        atom_lbound[1] + diameter,
                    )
                    start += extent
        else:
            # legacy behavior
            # Draw the atoms
            if (
                self.moving
                and obj_indx < len(self.move_atoms_mask)
                and self.move_atoms_mask[obj_indx]
            ):
                circle(
                    movecolor,
                    False,
                    atom_lbound[0] - 4,
                    atom_lbound[1] - 4,
                    atom_lbound[0] + diameter + 4,
                    atom_lbound[1] + diameter + 4,
                )

            circle(
                atom_color,
                selected,
                atom_lbound[0],
                atom_lbound[1],
                atom_lbound[0] + diameter,
                atom_lbound[1] + diameter,
                tags=("atom-circle", "atom-ghost") if ghost else ("atom-circle",),
            )

        # Draw labels on the atoms
        if self.labels is not None:
            if ghost and not self.ghost_settings["label"]:
                pass
            else:
                self.window.text(
                    atom_lbound[0] + diameter / 2,
                    atom_lbound[1] + diameter / 2,
                    str(self.labels[obj_indx]),
                )

    def draw(self, status=True):
        """Draw all required objects on the canvas.

        This is overridden from ``View``, in order to:

        - set bond colors, specific to the atom types at each end of the bond.
        - add a cross to 'ghost' atoms (specified by an array on the atoms)
        - add a dash pattern to the unit cell lines
        - allow miller index planes to be drawn

        """
        self.window.clear()

        # compute orientation, position and scale of axes
        axes = self.scale * self.axes * (1, -1, 1)
        offset = np.dot(self.center, axes)
        offset[:2] -= 0.5 * self.window.size

        # align elements to axes
        positions = {
            k: (
                np.dot(v._coordinates, axes)
                if v.etype == "sphere"
                else np.einsum("ijk, km -> ijm", v._coordinates, axes)
            )
            - offset
            for k, v in self.elements.items()
        }

        # compute atom radii
        atom_radii = self.get_covalent_radii() * self.scale
        if self.window["toggle-show-bonds"]:
            atom_radii *= 0.65
        # note self.P is used by View.release
        atom_xy = self.P = positions["atoms"][:, :2].round().astype(int)
        atom_lbound = (atom_xy - atom_radii[:, None]).round().astype(int)
        atom_diameters = (2 * atom_radii).round().astype(int)

        # work out the order to add elements in (z-order)
        # Note: for lines, we use the largest z of the start/end
        zorder_indices = self.indices = np.concatenate(
            (
                positions["atoms"] + atom_radii[:, None],
                positions["cell_lines"].max(axis=1),
                positions["bond_lines"].max(axis=1),
                positions["miller_lines"].max(axis=1),
                # the final plane coordinate can be np.nan, if it is a triangle
                np.nanmax(positions["miller_planes"], axis=1),
            )
        )[:, 2].argsort()

        # record the length of each element, so we can map to the zorder_indices
        num_atoms = len(positions["atoms"])
        num_cell_lines = len(positions["cell_lines"])
        num_bond_lines = len(positions["bond_lines"])
        num_miller_lines = len(positions["miller_lines"])

        # ensure coordinates are integer
        positions = {
            k: v.round().astype(int) if k != "miller_planes" else v
            for k, v in positions.items()
        }

        # compute other values necessary for drawing atoms
        atom_colors = self.get_colors()
        celldisp = (
            (np.dot(self.atoms.get_celldisp().reshape((3,)), axes)).round().astype(int)
        )
        constrained = ~self.images.get_dynamic(self.atoms)
        selected = self.images.selected
        visible = self.images.visible
        self.update_labels()  # set self.labels for atoms
        atom_tags = self.atoms.get_tags()  # extension for partial occupancies

        # extension for ghost atoms
        if "ghost" in self.atoms.arrays:
            ghost = self.atoms.get_array("ghost")
        else:
            ghost = [False for _ in self.atoms]

        # set linewidth for bonds
        bond_linewidth = self.scale * 0.15

        vector_arrays = []
        if self.window["toggle-show-velocities"]:
            # Scale ugly?
            vector = self.atoms.get_velocities()
            if vector is not None:
                vector_arrays.append(vector * 10.0 * self.velocity_vector_scale)
        if self.window["toggle-show-forces"]:
            f = self.get_forces()
            vector_arrays.append(f * self.force_vector_scale)

        for array in vector_arrays:
            array[:] = (np.dot(array, axes) + positions["atoms"]).round().astype(int)

        # setup drawing functions
        line = self.window.line
        if self.arrowkey_mode == self.ARROWKEY_MOVE:
            movecolor = GREEN
        elif self.arrowkey_mode == self.ARROWKEY_ROTATE:
            movecolor = PURPLE
        else:
            movecolor = None

        for obj_indx in zorder_indices:
            if obj_indx < num_atoms:

                if not visible[obj_indx]:
                    continue

                diameter = atom_diameters[obj_indx]
                # draw atom element
                self._draw_atom(
                    obj_indx,
                    diameter,
                    atom_lbound[obj_indx],
                    selected[obj_indx],
                    atom_colors[obj_indx],
                    ghost[obj_indx],
                    atom_tags[obj_indx],
                    movecolor,
                )

                # Draw cross on constrained or ghost atoms
                if constrained[obj_indx] or (
                    ghost[obj_indx] and self.ghost_settings["cross"]
                ):
                    rad1 = int(0.14644 * diameter)
                    rad2 = int(0.85355 * diameter)
                    line(
                        (
                            atom_lbound[obj_indx, 0] + rad1,
                            atom_lbound[obj_indx, 1] + rad1,
                            atom_lbound[obj_indx, 0] + rad2,
                            atom_lbound[obj_indx, 1] + rad2,
                        )
                    )
                    line(
                        (
                            atom_lbound[obj_indx, 0] + rad2,
                            atom_lbound[obj_indx, 1] + rad1,
                            atom_lbound[obj_indx, 0] + rad1,
                            atom_lbound[obj_indx, 1] + rad2,
                        )
                    )

                # Draw velocities and/or forces
                for vector in vector_arrays:
                    assert not np.isnan(vector).any()
                    self.arrow(
                        (
                            positions["atoms"][obj_indx, 0],
                            positions["atoms"][obj_indx, 1],
                            vector[obj_indx, 0],
                            vector[obj_indx, 1],
                        ),
                        width=2,
                    )
            elif obj_indx < num_atoms + num_cell_lines:
                # Draw unit cell lines
                line_idx = obj_indx - num_atoms
                self.window.canvas.create_line(
                    (
                        positions["cell_lines"][line_idx, 0, 0] + celldisp[0],
                        positions["cell_lines"][line_idx, 0, 1] + celldisp[1],
                        positions["cell_lines"][line_idx, 1, 0] + celldisp[0],
                        positions["cell_lines"][line_idx, 1, 1] + celldisp[1],
                    ),
                    width=1,
                    dash=(6, 4),  # dash pattern = (line length, gap length, ..)
                    tags=("cell-line",),
                )
            elif obj_indx < num_atoms + num_cell_lines + num_bond_lines:
                # Draw bond lines, splitting in half, and colouring each half by nearest atom
                # TODO would be nice if bonds had an outline
                line_idx = obj_indx - num_atoms - num_cell_lines
                start_atom, end_atom = self.elements["bond_lines"][line_idx].atom_index
                self.window.canvas.create_line(
                    (
                        positions["bond_lines"][line_idx, 0, 0],
                        positions["bond_lines"][line_idx, 0, 1],
                        positions["bond_lines"][line_idx, 0, 0]
                        + 0.5
                        * (
                            positions["bond_lines"][line_idx, 1, 0]
                            - positions["bond_lines"][line_idx, 0, 0]
                        ),
                        positions["bond_lines"][line_idx, 0, 1]
                        + 0.5
                        * (
                            positions["bond_lines"][line_idx, 1, 1]
                            - positions["bond_lines"][line_idx, 0, 1]
                        ),
                    ),
                    width=bond_linewidth,
                    fill=atom_colors[start_atom],
                    tags=("bond-line",),
                )
                self.window.canvas.create_line(
                    (
                        positions["bond_lines"][line_idx, 0, 0]
                        + 0.5
                        * (
                            positions["bond_lines"][line_idx, 1, 0]
                            - positions["bond_lines"][line_idx, 0, 0]
                        ),
                        positions["bond_lines"][line_idx, 0, 1]
                        + 0.5
                        * (
                            positions["bond_lines"][line_idx, 1, 1]
                            - positions["bond_lines"][line_idx, 0, 1]
                        ),
                        positions["bond_lines"][line_idx, 1, 0],
                        positions["bond_lines"][line_idx, 1, 1],
                    ),
                    width=bond_linewidth,
                    fill=atom_colors[end_atom],
                    tags=("bond-line",),
                )
            elif (
                obj_indx
                < num_atoms + num_cell_lines + num_bond_lines + num_miller_lines
            ):
                miller_indx = obj_indx - num_atoms - num_cell_lines - num_bond_lines
                self.window.canvas.create_line(
                    (
                        positions["miller_lines"][miller_indx, 0, 0] + celldisp[0],
                        positions["miller_lines"][miller_indx, 0, 1] + celldisp[1],
                        positions["miller_lines"][miller_indx, 1, 0] + celldisp[0],
                        positions["miller_lines"][miller_indx, 1, 1] + celldisp[1],
                    ),
                    width=self.get_miller_thickness(
                        self.elements["miller_lines"][miller_indx].index
                    ),
                    fill=self.get_miller_color(
                        self.elements["miller_lines"][miller_indx].index
                    ),
                    tags=("miller-line",),
                )
            else:
                miller_indx = (
                    obj_indx
                    - num_atoms
                    - num_cell_lines
                    - num_bond_lines
                    - num_miller_lines
                )
                plane = positions["miller_planes"][miller_indx]
                plane_pts = [
                    pt[i] + celldisp[i]
                    for pt in plane.round().astype(int)
                    for i in [0, 1]
                ]
                self.window.canvas.create_polygon(
                    plane_pts,
                    width=self.get_miller_thickness(miller_indx),
                    outline=self.get_miller_color(miller_indx),
                    fill=self.get_miller_color(miller_indx),
                    tags=("miller-plane",),
                )

        if self.window["toggle-show-axes"]:
            self.draw_axes()

        if len(self.images) > 1:
            self.draw_frame_number()

        self.window.update()

        if status:
            self.status(self.atoms)
