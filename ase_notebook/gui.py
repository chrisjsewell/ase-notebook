"""A module for visualising structures.

The module subclasses ase (v3.18.0) classes, to add additional functionality.
"""
from time import time
import tkinter
from tkinter.font import Font

from ase.data import atomic_numbers
from ase.data import covalent_radii as default_covalent_radii
from ase.gui import ui
from ase.gui.gui import GUI
from ase.gui.images import Images
from ase.gui.view import GREEN, PURPLE, View
import attr
import numpy as np

from aiida_2d.visualize.atom_info import create_info_lines
from aiida_2d.visualize.color import lighten_webcolor
from aiida_2d.visualize.core import initialise_element_groups


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

    def __init__(self, config, images=None, element_colors=None):
        """Initialise the GUI.

        Parameters
        ----------
        config: ViewConfig
            initial configuration settings
        images : ase.gui.images.Images
            list of ase.Atoms, with some settings for visualisations (mainly radii)
        element_colors: list[tuple]
            hex colour for each atomic number (defaults to 'jmol' scheme)

        """
        if not isinstance(images, Images):
            images = Images(images)

        self.images = images
        self.observers = []

        self.config = attr.asdict(config)

        # aliases required by ui.ASEGUIWindow
        self.config["gui_foreground_color"] = self.config["canvas_color_foreground"]
        self.config["gui_background_color"] = self.config["canvas_color_background"]
        self.config["swap_mouse"] = self.config["gui_swap_mouse"]

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
        # used by ``View.update_labels``
        label_sites = {"index": 1, "magmom": 2, "element": 3, "charge": 4}.get(
            self.config["atom_label_by"], 0
        )
        if not self.config["atom_show_label"]:
            label_sites = 0
        self.window["show-labels"] = label_sites

        View.__init__(self, self.config["rotations"])
        if element_colors:
            self.colors = dict(enumerate(element_colors))

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

    def move(self, event):
        """Handle move event."""
        # fix an error raised in GUI class
        if self.xy is None:
            self.xy = (event.x, event.y)

        super().move(event)

    def showing_millers(self):
        """Return whether to display planes."""
        return self.config["show_miller_planes"]

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
            show_unit_cell=self.showing_cell(),
            uc_dash_pattern=self.config["uc_dash_pattern"],
            show_bonds=self.showing_bonds(),
            bond_supercell=self.images.repeat,
            miller_planes=self.config["miller_planes"]
            if self.showing_millers()
            else None,
            miller_planes_as_lines=self.config["miller_as_lines"],
        )
        self.elements = elements

        # record all positions (atoms first) with legacy array name, for use by View.focus
        self.X = elements.get_all_coordinates()
        # record atom positions with legacy array name, used by View.move
        self.X_pos = self.elements["atoms"]._positions.copy()

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

        element_groups = self.elements
        element_groups.update_positions(
            axes,
            offset,
            radii_scale=self.scale * 0.65
            if self.window["toggle-show-bonds"]
            else self.scale,
        )

        # required by View.release
        self.P = element_groups["atoms"].unstack_positions()[:, :2].round().astype(int)
        self.indices = np.array([i for i, _ in element_groups.yield_zorder()])

        if "ghost" in self.atoms.arrays:
            ghost_atoms = self.atoms.get_array("ghost")
        else:
            ghost_atoms = [False for _ in self.atoms]

        self.update_labels()  # set self.labels for atoms
        # TODO use occuapancy keys for label, if showing symbol

        atom_colors = self.get_colors()

        if self.config["atom_lighten_by_depth"]:
            z_positions = element_groups["atoms"].get_max_zposition()
            zmin, zmax = z_positions.min(), z_positions.max()
            new_atom_colors = []
            for atom_color, z_position in zip(atom_colors, z_positions):
                atom_depth = (zmax - z_position) / (zmax - zmin)
                atom_color = lighten_webcolor(
                    atom_color, atom_depth * self.config["atom_lighten_by_depth"]
                )
                new_atom_colors.append(atom_color)
            atom_colors = new_atom_colors

        celldisp = (
            (np.dot(self.atoms.get_celldisp().reshape((3,)), axes)).round().astype(int)
        )

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
            array[:] = (
                (np.dot(array, axes) + element_groups["atoms"].unstack_positions())
                .round()
                .astype(int)
            )

        element_groups["atoms"].set_property_many(
            {
                "lbound": (
                    element_groups["atoms"].unstack_positions()[:, :2]
                    - element_groups["atoms"].scaled_radii[:, None]
                )
                .round()
                .astype(int),
                "color": atom_colors,
                "label": [
                    None if g and not self.config["ghost_show_label"] else l
                    for l, g in zip(
                        self.labels or [None for _ in ghost_atoms], ghost_atoms
                    )
                ],
                "tag": self.atoms.get_tags(),
                "ghost": ghost_atoms,
                "selected": self.images.selected,
                "visible": self.images.visible,
                "constrained": ~self.images.get_dynamic(self.atoms),
                "moving": [m if self.moving else False for m in self.move_atoms_mask]
                if self.move_atoms_mask is not None
                else [False for _ in self.atoms],
                "occupancy": [
                    self.atoms.info["occupancy"][t]
                    if "occupancy" in self.atoms.info
                    else None
                    for t in self.atoms.get_tags()
                ],
                "stroke_width": [
                    self.config["ghost_stroke_width"]
                    if g
                    else self.config["atom_stroke_width"]
                    for g in ghost_atoms
                ],
            },
            element=True,
        )
        element_groups["atoms"].set_property_many(
            {
                "font_size": self.config["atom_font_size"],
                "font_color": self.config["atom_font_color"],
            },
            element=False,
        )
        element_groups["cell_lines"].set_property_many(
            {"color": self.config["uc_color"]}, element=False
        )
        element_groups["bond_lines"].set_property(
            "color",
            [
                (atom_colors[i], atom_colors[j])
                for i, j in element_groups["bond_lines"].get_elements_property(
                    "atom_index"
                )
            ],
            element=True,
        )
        element_groups["bond_lines"].set_property_many(
            {"stroke_width": self.scale * 0.15}, element=False
        )
        for miller_type in ["miller_lines", "miller_planes"]:
            element_groups[miller_type].set_property_many(
                {
                    "fill_color": [
                        self.config["miller_planes"][i].get("fill_color", "blue")
                        for i in element_groups[miller_type].get_elements_property(
                            "index"
                        )
                    ],
                    "stroke_color": [
                        self.config["miller_planes"][i].get("stroke_color", "blue")
                        for i in element_groups[miller_type].get_elements_property(
                            "index"
                        )
                    ],
                    "stroke_width": [
                        self.config["miller_planes"][i].get("stroke_width", 1)
                        for i in element_groups[miller_type].get_elements_property(
                            "index"
                        )
                    ],
                },
                element=True,
            )

        if self.arrowkey_mode == self.ARROWKEY_MOVE:
            movecolor = GREEN
        elif self.arrowkey_mode == self.ARROWKEY_ROTATE:
            movecolor = PURPLE
        else:
            movecolor = None

        draw_elements(
            element_groups,
            canvas=self.window.canvas,
            celldisp=celldisp,
            vector_arrays=vector_arrays,
            movecolor=movecolor,
            scale=self.scale,
            element_colors=self.colors,
            ghost_cross_out=self.config["ghost_cross_out"],
            background_color=self.config["gui_background_color"],
        )

        if self.window["toggle-show-axes"]:
            draw_axes(
                self.window.canvas,
                self.axes,
                self.window.size,
                length=self.config["axes_length"],
                font_size=self.config["axes_font_size"],
                line_color=self.config["axes_line_color"],
            )

        if len(self.images) > 1:
            self.draw_frame_number()

        self.window.update()

        if status:
            num_atoms = len(self.atoms)
            indices = np.arange(num_atoms)[self.images.selected[:num_atoms]]
            ordered_indices = [i for i in self.images.selected_ordered if i < num_atoms]
            status_lines = create_info_lines(self.atoms, indices, ordered_indices)
            self.window.update_status_line(" " + "; ".join(status_lines))


def draw_arrow(canvas, coords, width, scale):
    """Draw an arrow element."""
    begin = np.array((coords[0], coords[1]))
    end = np.array((coords[2], coords[3]))
    canvas.create_line(*tuple(int(x) for x in coords), width)

    vec = end - begin
    length = np.sqrt((vec[:2] ** 2).sum())
    length = min(length, 0.3 * scale)

    angle = np.arctan2(end[1] - begin[1], end[0] - begin[0]) + np.pi
    x1 = (end[0] + length * np.cos(angle - 0.3)).round().astype(int)
    y1 = (end[1] + length * np.sin(angle - 0.3)).round().astype(int)
    x2 = (end[0] + length * np.cos(angle + 0.3)).round().astype(int)
    y2 = (end[1] + length * np.sin(angle + 0.3)).round().astype(int)
    canvas.create_line(x1, y1, int(end[0]), int(end[1]), width)
    canvas.create_line(x2, y2, int(end[0]), int(end[1]), width)


def draw_circle(lbound, diameter, canvas, color, selected, tags=(), stroke_width=1):
    """Draw a circle element, given a lower bound and diameter."""
    if selected:
        outline = "#004500"
        width = stroke_width * 3
    else:
        outline = "black"
        width = stroke_width

    bbox = (lbound[0], lbound[1], lbound[0] + diameter, lbound[1] + diameter)

    canvas.create_oval(
        *tuple(int(x) for x in bbox),
        fill=color,
        outline=outline,
        width=width,
        tags=tags,
    )


def draw_arc(lbound, diameter, canvas, color, selected, start, extent):
    """Draw an arc element."""
    if selected:
        outline = "#004500"
        width = 3
    else:
        outline = "black"
        width = 1
    bbox = (lbound[0], lbound[1], lbound[0] + diameter, lbound[1] + diameter)
    canvas.create_arc(
        *tuple(int(x) for x in bbox),
        start=start,
        extent=extent,
        fill=color,
        outline=outline,
        width=width,
    )


def draw_axes(
    canvas,
    axes,
    window_size,
    *,
    length=15,
    line_color="black",
    line_width=1,
    font_size=14,
):
    """Draw the axes element."""
    rgb = ["red", "green", "blue"]

    for i in axes[:, 2].argsort():
        a = 20
        b = window_size[1] - 20
        c = int(axes[i][0] * length + a)
        d = int(-axes[i][1] * length + b)
        canvas.create_line(
            *tuple(int(x) for x in (a, b, c, d)), width=line_width, fill=line_color
        )
        canvas.create_text(
            (c, d),
            text="XYZ"[i],
            fill=rgb[i],
            font=Font(size=20),
            anchor=tkinter.CENTER,
        )


def draw_elements(
    element_groups,
    canvas,
    celldisp,
    vector_arrays,
    scale,
    element_colors,
    ghost_cross_out=False,
    movecolor=None,
    background_color="#ffffff",
):
    """Draw elements on a ``tkinter.Canvas``.

    Parameters
    ----------
    element_groups : aiida_2d.visualize.core.DrawGroup
    canvas : tkinter.Canvas
    celldisp : numpy.array
        cell displacement
    vector_arrays : list
    scale : float
        canvas scale (used for drawing vector arrows)
    element_colors : dict
        mapping of element colors (used for partial occupancies)
    ghost_cross_out : bool
        whether to cross out ghost atoms
    movecolor : str or None
        color for moving atom
    background_color : str or None
        color used for coloring partial atom occupancies

    """
    for idx, element in element_groups.yield_zorder():
        if element.name == "atoms":

            if not element.visible:
                continue

            diameter = int(round(element.sradius * 2))
            if element.occupancy is not None:
                # first draw an empty circle if a site is not fully occupied
                if (np.sum([o for o in element.occupancy.values()])) < 1.0:
                    draw_circle(
                        element.lbound,
                        diameter,
                        canvas,
                        background_color,
                        element.selected,
                        stroke_width=element.stroke_width,
                    )
                start = 0
                # start with the dominant species
                for sym, occ in sorted(
                    element.occupancy.items(), key=lambda x: x[1], reverse=True
                ):
                    if np.round(occ, decimals=4) == 1.0:
                        draw_circle(
                            element.lbound,
                            diameter,
                            canvas,
                            element_colors[atomic_numbers[sym]],
                            element.selected,
                            stroke_width=element.stroke_width,
                            tags=("atom-circle",),
                        )
                    else:
                        # TODO alter for ghost
                        extent = 360.0 * occ
                        draw_arc(
                            element.lbound,
                            diameter,
                            canvas,
                            element_colors[atomic_numbers[sym]],
                            element.selected,
                            start,
                            extent,
                        )
                        start += extent
            else:
                if element.moving:
                    draw_circle(
                        (element.lbound[0] - 4, element.lbound[1] - 4),
                        diameter + 8,
                        canvas,
                        movecolor,
                        False,
                    )
                draw_circle(
                    element.lbound,
                    diameter,
                    canvas,
                    element.color,
                    element.selected,
                    stroke_width=element.stroke_width,
                    tags=("atom-circle",),
                )

            if element.label is not None:
                canvas.create_text(
                    (
                        element.lbound[0] + diameter / 2,
                        element.lbound[1] + diameter / 2,
                    ),
                    text=str(element.label),
                    fill=element.font_color,
                    font=Font(size=element.font_size),
                    anchor=tkinter.CENTER,
                )

            # Draw cross on constrained or ghost atoms
            if element.constrained or (element.ghost and ghost_cross_out):
                rad1 = int(0.14644 * diameter)
                rad2 = int(0.85355 * diameter)
                canvas.create_line(
                    element.lbound[0] + rad1,
                    element.lbound[1] + rad1,
                    element.lbound[0] + rad2,
                    element.lbound[1] + rad2,
                    width=1,
                )
                canvas.create_line(
                    element.lbound[0] + rad2,
                    element.lbound[1] + rad1,
                    element.lbound[0] + rad1,
                    element.lbound[1] + rad2,
                    width=1,
                )

            # Draw velocities and/or forces
            # TODO vector data should be added to element
            for vector in vector_arrays:
                assert not np.isnan(vector).any()
                draw_arrow(
                    canvas(
                        element.position[0],
                        element.position[1],
                        vector[idx, 0],
                        vector[idx, 1],
                    ),
                    width=2,
                    scale=scale,
                )

        if element.name == "cell_lines":

            canvas.create_line(
                (
                    element.position[0, 0] + celldisp[0],
                    element.position[0, 1] + celldisp[1],
                    element.position[1, 0] + celldisp[0],
                    element.position[1, 1] + celldisp[1],
                ),
                fill=element.color,
                width=1,
                # dash=(6, 4),  # dash pattern = (line length, gap length, ..)
                tags=("cell-line",),
            )

        if element.name == "bond_lines":

            canvas.create_line(
                (
                    element.position[0, 0],
                    element.position[0, 1],
                    element.position[0, 0]
                    + 0.5 * (element.position[1, 0] - element.position[0, 0]),
                    element.position[0, 1]
                    + 0.5 * (element.position[1, 1] - element.position[0, 1]),
                ),
                width=element.stroke_width,
                fill=element.color[0],
                tags=("bond-line",),
            )
            canvas.create_line(
                (
                    element.position[0, 0]
                    + 0.5 * (element.position[1, 0] - element.position[0, 0]),
                    element.position[0, 1]
                    + 0.5 * (element.position[1, 1] - element.position[0, 1]),
                    element.position[1, 0],
                    element.position[1, 1],
                ),
                width=element.stroke_width,
                fill=element.color[1],
                tags=("bond-line",),
            )

        if element.name == "miller_lines":

            canvas.create_line(
                (
                    element.position[0, 0] + celldisp[0],
                    element.position[0, 1] + celldisp[1],
                    element.position[1, 0] + celldisp[0],
                    element.position[1, 1] + celldisp[1],
                ),
                width=element.stroke_width,
                fill=element.stroke_color,
                tags=("miller-line",),
            )

        if element.name == "miller_planes":

            plane_pts = [
                pt[i] + celldisp[i]
                for pt in element.position.round().astype(int)
                for i in [0, 1]
            ]
            canvas.create_polygon(
                plane_pts,
                width=element.stroke_width,
                outline=element.stroke_color,
                fill=element.fill_color,
                tags=("miller-plane",),
            )
