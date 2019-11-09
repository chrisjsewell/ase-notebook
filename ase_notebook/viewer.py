"""A module for creating visualisations of a structure."""
import inspect
import json
import subprocess
import sys
from time import sleep
from typing import Union

from ase.data import covalent_radii as ase_covalent_radii
from ase.data.colors import jmol_colors as ase_element_colors
import attr
import numpy as np

from .atom_info import create_info_lines
from .atoms_convert import convert_to_atoms, deserialize_atoms, serialize_atoms
from .backend.gui import AtomGui, AtomImages
from .backend.svg import (
    create_axes_elements,
    create_svg_document,
    generate_svg_elements,
)
from .backend.threejs import (
    create_world_axes,
    generate_3js_render,
    make_basic_gui,
    RenderContainer,
)
from .color import lighten_webcolor
from .configuration import ViewConfig
from .data import load_data_file
from .draw_utils import (
    compute_projection,
    get_rotation_matrix,
    initialise_element_groups,
)


class AseView:
    """Class for visualising ``ase.Atoms`` or ``pymatgen.Structure``."""

    def __init__(self, config: Union[None, ViewConfig] = None, **kwargs):
        """This is replaced by ``SVGConfig`` docstring."""  # noqa: D401
        if config is not None:
            self._config = config
        else:
            self._config = ViewConfig(**kwargs)

    @property
    def config(self):
        """Return the visualisation configuration."""
        return self._config

    def get_config_as_dict(self):
        """Return the configuration as a JSONable dictionary."""
        return attr.asdict(self._config)

    def get_input_as_dict(self, atoms, **kwargs):
        """Return the configuration, atoms and kwargs as a JSONable dictionary."""
        return {
            "config": attr.asdict(self._config),
            "atoms": serialize_atoms(convert_to_atoms(atoms)),
            "kwargs": kwargs,
        }

    def add_miller_plane(
        self,
        h,
        k,
        l,
        *,
        color="blue",
        stroke_color=None,
        stroke_width=1,
        fill_opacity=0.5,
        stroke_opacity=0.9,
        reset=False,
    ):
        """Add a miller plane to the config.

        Parameters
        ----------
        h : int or float
        k : int or float
        l : int or float
        color : str
            color of plane
        stroke_color : str or None
            color of outline (if None, color is used)
        stroke_width : int
            width of outline
        reset : bool
            if True, remove any previously set miller planes

        """
        plane = [
            {
                "h": h,
                "k": k,
                "l": l,
                "fill_color": color,
                "stroke_color": stroke_color or color,
                "stroke_width": stroke_width,
                "fill_opacity": fill_opacity,
                "stroke_opacity": stroke_opacity,
            }
        ]
        if reset:
            self.config.miller_planes = plane
        else:
            self.config.miller_planes = list(self.config.miller_planes) + plane

    def get_element_colors(self):
        """Return mapping of element atomic number to (hex) color."""
        if self.config.element_colors == "ase":
            return [
                "#{0:02X}{1:02X}{2:02X}".format(*(int(x * 255) for x in c))
                for c in ase_element_colors
            ]
        if self.config.element_colors == "vesta":
            data = load_data_file("vesta_element_data.json")
            return [
                "#{0:02X}{1:02X}{2:02X}".format(*(int(x * 255) for x in (r, g, b)))
                for r, g, b in zip(data["r"], data["g"], data["b"])
            ]
        raise ValueError(self.config.element_colors)

    def get_element_radii(self):
        """Return mapping of element atomic number to atom radii."""
        if self.config.element_radii == "ase":
            return ase_covalent_radii.copy()
        if self.config.element_radii == "vesta":
            data = load_data_file("vesta_element_data.json")
            return data["radius"]
        raise ValueError(self.config.element_radii)

    def get_atom_colors(self, atoms):
        """Return mapping of atom index to (hex) color."""
        if self.config.atom_color_by == "element":
            element_colors = self.get_element_colors()
            return [element_colors[z] for z in atoms.numbers]
        if self.config.atom_color_by == "color_array":
            return atoms.get_array(self.config.atom_color_array)

        if self.config.atom_color_by == "index":
            values = range(len(atoms))
        elif self.config.atom_color_by == "tag":
            values = atoms.get_tags()
        elif self.config.atom_color_by == "magmom":
            values = atoms.get_initial_magnetic_moments()
        elif self.config.atom_color_by == "charge":
            values = atoms.get_initial_charges()
        elif self.config.atom_color_by == "velocity":
            values = (atoms.get_velocities() ** 2).sum(1) ** 0.5
        elif self.config.atom_color_by == "value_array":
            values = atoms.get_array(self.config.atom_color_array)
        else:
            raise ValueError(self.config.atom_color_by)

        from matplotlib.cm import get_cmap
        from matplotlib.colors import Normalize, rgb2hex

        cmap = get_cmap(self.config.atom_colormap)
        cmin, cmax = self.config.atom_colormap_range
        norm = Normalize(
            vmin=min(values) if cmin is None else cmin,
            vmax=max(values) if cmax is None else cmax,
        )
        return [rgb2hex(cmap(norm(v))[:3]) for v in values]

    def get_atom_radii(self, atoms):
        """Return mapping of atom index to sphere radii."""
        element_radii = self.get_element_radii()
        radii = np.array([element_radii[z] for z in atoms.numbers])
        radii *= self.config.radii_scale
        return radii

    def get_atom_labels(self, atoms):
        """Return mapping of atom index to text label."""
        labels = None
        if self.config.atom_label_by == "element":
            if "occupancy" in atoms.info:
                labels = [
                    ",".join(atoms.info["occupancy"][t].keys())
                    for t in atoms.get_tags()
                ]
            else:
                labels = atoms.get_chemical_symbols()
        elif self.config.atom_label_by == "index":
            labels = list(range(len(atoms)))
        elif self.config.atom_label_by == "tag":
            labels = atoms.get_tags()
        elif self.config.atom_label_by == "magmom":
            labels = atoms.get_initial_magnetic_moments()
        elif self.config.atom_label_by == "charge":
            labels = atoms.get_initial_charges()
        elif self.config.atom_label_by == "array":
            labels = atoms.get_array(self.config.atom_label_array)

        if labels is None:
            raise ValueError(self.config.atom_label_by)

        return [str(l) for l in labels]

    def _initialise_elements(self, atoms, center_in_uc=False, repeat_uc=(1, 1, 1)):
        """Prepare visualisation elements, in a backend agnostic manner."""
        config = self._config

        atoms = convert_to_atoms(atoms)

        if center_in_uc:
            atoms.center()
        atoms = atoms.repeat(repeat_uc)
        if config.show_uc_repeats:
            atoms.info["unit_cell_repeat"] = (
                repeat_uc
                if isinstance(config.show_uc_repeats, bool)
                else config.show_uc_repeats
            )

        element_groups = initialise_element_groups(
            atoms,
            atom_radii=self.get_atom_radii(atoms),
            show_unit_cell=config.show_unit_cell,
            uc_dash_pattern=config.uc_dash_pattern,
            show_bonds=config.show_bonds,
            miller_planes=config.miller_planes if config.show_miller_planes else None,
            miller_planes_as_lines=config.miller_as_lines,
        )

        return atoms, element_groups

    def _add_element_properties(
        self, atoms, element_groups, bond_thickness, lighten_by_depth=True
    ):
        """Add initial properties to the element groups."""
        config = self._config

        atom_colors = self.get_atom_colors(atoms)
        atom_labels = self.get_atom_labels(atoms)
        ghost_atoms = (
            atoms.get_array("ghost")
            if "ghost" in atoms.arrays
            else [False for _ in atoms]
        )

        if config.atom_lighten_by_depth and lighten_by_depth:
            z_positions = element_groups["atoms"].get_max_zposition()
            zmin, zmax = z_positions.min(), z_positions.max()
            new_atom_colors = []
            for atom_color, z_position in zip(atom_colors, z_positions):
                atom_depth = (zmax - z_position) / (zmax - zmin)
                atom_color = lighten_webcolor(
                    atom_color, atom_depth * config.atom_lighten_by_depth
                )
                new_atom_colors.append(atom_color)
            atom_colors = new_atom_colors

        element_groups["atoms"].set_property_many(
            {
                "color": atom_colors,
                "label": [
                    None if g or not config.atom_show_label else l
                    for l, g in zip(atom_labels, ghost_atoms)
                ],
                "ghost": ghost_atoms,
                "fill_opacity": [
                    config.ghost_opacity if g else config.atom_opacity
                    for g in ghost_atoms
                ],
                "occupancy": [
                    atoms.info["occupancy"][t] if "occupancy" in atoms.info else None
                    for t in atoms.get_tags()
                ],
                "stroke_width": [
                    config.ghost_stroke_width if g else config.atom_stroke_width
                    for g in ghost_atoms
                ],
                "stroke_opacity": [
                    config.ghost_stroke_opacity if g else config.atom_stroke_opacity
                    for g in ghost_atoms
                ],
                "info_string": [
                    "; ".join(create_info_lines(atoms, [i])) for i in range(len(atoms))
                ],
            },
            element=True,
        )
        element_groups["atoms"].set_property_many(
            {"font_size": config.atom_font_size, "font_color": config.atom_font_color},
            element=False,
        )
        element_groups["cell_lines"].set_property_many(
            {"color": config.uc_color}, element=False
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
            {"stroke_width": bond_thickness, "stroke_opacity": config.bond_opacity},
            element=False,
        )
        for miller_type in ["miller_lines", "miller_planes"]:
            element_groups[miller_type].set_property_many(
                {
                    "fill_color": [
                        config.miller_planes[i].get("fill_color", "blue")
                        for i in element_groups[miller_type].get_elements_property(
                            "index"
                        )
                    ],
                    "stroke_color": [
                        config.miller_planes[i].get("stroke_color", "blue")
                        for i in element_groups[miller_type].get_elements_property(
                            "index"
                        )
                    ],
                    "stroke_width": [
                        config.miller_planes[i].get("stroke_width", 1)
                        for i in element_groups[miller_type].get_elements_property(
                            "index"
                        )
                    ],
                    "fill_opacity": [
                        config.miller_planes[i].get("fill_opacity", 1)
                        for i in element_groups[miller_type].get_elements_property(
                            "index"
                        )
                    ],
                    "stroke_opacity": [
                        config.miller_planes[i].get("stroke_opacity", 1)
                        for i in element_groups[miller_type].get_elements_property(
                            "index"
                        )
                    ],
                },
                element=True,
            )

    def make_svg(self, atoms, center_in_uc=False, repeat_uc=(1, 1, 1)):
        """Create an SVG of the atoms or structure."""
        config = self.config
        atoms, element_groups = self._initialise_elements(
            atoms, center_in_uc=center_in_uc, repeat_uc=repeat_uc
        )

        rotation_matrix = get_rotation_matrix(config.rotations)

        center, scale = compute_projection(
            element_groups, config.canvas_size, rotation_matrix
        )
        scale *= config.zoom
        axes = scale * rotation_matrix * (1, -1, 1)
        offset = np.dot(center, axes)
        offset[:2] -= 0.5 * np.array(config.canvas_size)
        element_groups.update_positions(
            axes, offset, radii_scale=scale * 0.65 if config.show_bonds else scale
        )

        self._add_element_properties(atoms, element_groups, bond_thickness=scale * 0.15)

        svg_elements = generate_svg_elements(
            element_groups,
            element_colors=self.get_element_colors(),
            background_color=config.canvas_color_background,
        )

        if config.canvas_crop:
            left, right, top, bottom = config.canvas_crop
            # (left, right, top, bottom) -> (minx, miny, width, height)
            viewbox = (
                left,
                top,
                config.canvas_size[0] - left - right,
                config.canvas_size[1] - top - bottom,
            )
        else:
            left = right = top = bottom = 0
            viewbox = (0, 0, config.canvas_size[0], config.canvas_size[1])

        if config.show_axes:
            svg_elements.extend(
                create_axes_elements(
                    rotation_matrix * (1, -1, 1),
                    config.canvas_size,
                    inset=(20 + left, 20 + bottom),
                    length=config.axes_length,
                    font_size=config.axes_font_size,
                    line_color=config.axes_line_color,
                )
            )

        return create_svg_document(
            svg_elements,
            config.canvas_size,
            viewbox if config.canvas_crop else None,
            background_color=config.canvas_color_background,
            background_opacity=config.canvas_background_opacity,
        )

    def make_gui(
        self,
        atoms,
        center_in_uc=False,
        repeat_uc=(1, 1, 1),
        bring_to_top=True,
        launch=True,
    ):
        """Launch a (blocking) GUI to view the atoms or structure."""
        atoms, element_groups = self._initialise_elements(
            atoms, center_in_uc=center_in_uc, repeat_uc=repeat_uc
        )

        images = AtomImages(
            [atoms],
            element_radii=np.array(self.get_element_radii()).tolist(),
            radii_scale=self.config.radii_scale,
        )
        gui = AtomGui(
            config=self.config, images=images, element_colors=self.get_element_colors()
        )
        if bring_to_top:
            tk_window = gui.window.win  # tkinter.Toplevel
            tk_window.attributes("-topmost", 1)
            tk_window.attributes("-topmost", 0)
        if launch:
            gui.run()
        else:
            return gui

    def launch_gui_subprocess(
        self, atoms, center_in_uc=False, repeat_uc=(1, 1, 1), test_init=2
    ):
        """Launch a GUI to view the atoms or structure, in a (non-blocking) subprocess.

        We encode all the data into a json object,
        then parse this to a console executable via stdin.

        :param test_init: wait for a x seconds, then test whether the process initialized without error.

        """
        atoms = convert_to_atoms(atoms)
        data_str = json.dumps(
            {
                "atoms": serialize_atoms(atoms),
                "config": self.get_config_as_dict(),
                "kwargs": {"center_in_uc": center_in_uc, "repeat_uc": repeat_uc},
            }
        )
        process = subprocess.Popen(
            "ase-notebook.view_atoms", stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.stdin.write(data_str.encode())
        process.stdin.close()
        sleep(test_init)
        if process.poll():
            raise RuntimeError(process.stderr.read().decode())
        return process

    def make_render(
        self,
        atoms,
        center_in_uc=False,
        repeat_uc=(1, 1, 1),
        reuse_objects=True,
        use_atom_arrays=False,
        use_label_arrays=True,
        create_gui=True,
    ):
        """Create a pythreejs render of the atoms or structure."""
        config = self.config
        atoms, element_groups = self._initialise_elements(
            atoms, center_in_uc=center_in_uc, repeat_uc=repeat_uc
        )
        rotation_matrix = get_rotation_matrix(config.rotations)
        element_groups.update_positions(axes=rotation_matrix)

        pos_min, pos_max = element_groups.get_position_range()

        element_groups.update_positions(
            axes=rotation_matrix,
            offset=pos_min + (pos_max - pos_min) / 2,
            radii_scale=0.65 if config.show_bonds else 1.0,
        )

        self._add_element_properties(
            atoms, element_groups, bond_thickness=5, lighten_by_depth=False
        )

        renderer, key_elements = generate_3js_render(
            element_groups,
            canvas_size=config.canvas_size,
            zoom=config.zoom,
            background_color=config.canvas_color_background,
            background_opacity=config.canvas_background_opacity,
            camera_fov=config.camera_fov,
            reuse_objects=reuse_objects,
            use_atom_arrays=use_atom_arrays,
            use_label_arrays=use_label_arrays,
        )
        if config.show_axes:
            axes_renderer = create_world_axes(
                renderer.camera, renderer.controls[0], initial_rotation=rotation_matrix
            )
            key_elements["axes_renderer"] = axes_renderer

        container = RenderContainer(renderer, element_renderer=renderer, **key_elements)
        if create_gui:
            gui = make_basic_gui(container)
            container.top_level = gui

        return container


AseView.__init__.__doc__ = (
    "kwargs are used to initialise ViewConfig:"
    f"\n{inspect.signature(ViewConfig.__init__)}"
)


def launch_gui_exec(json_string=None):
    """Launch a GUI, with a json string as input.

    Parameters
    ----------
    json_string : str or None
        A json string containing all data required for running AseView.makegui.
        If None, the string is read from ``stdin``.

    """
    if json_string is None:
        if sys.stdin.isatty():
            raise IOError("stdin is empty")
        json_string = sys.stdin.read()
    data = json.loads(json_string)
    atoms_json = data.pop("atoms", {})
    config_dict = data.pop("config", {})
    kwargs = data.pop("kwargs", {})

    atoms = deserialize_atoms(atoms_json)
    ase_view = AseView(**config_dict)
    return ase_view.make_gui(atoms, **kwargs)


# Note: original commands (when creating SVG via tkinter postscript)
# gui.window.win.withdraw()  # hide window
# canvas = gui.window.canvas
# canvas.config(width=100, height=100); gui.draw()  # resize canvas
# gui.scale *= zoom; gui.draw()  # zoom
# canvas.postscript(file=fname)  # save canvas
