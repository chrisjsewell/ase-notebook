"""A module for creating visualisations of a structure."""
import importlib
import inspect
import json
import subprocess
import sys
from time import sleep
from typing import Union

import ase
from ase.data import covalent_radii as ase_covalent_radii
from ase.data.colors import jmol_colors as ase_element_colors
import attr
from attr.validators import in_, instance_of
import jsonschema
import numpy as np

from aiida_2d.visualize.color import Color, lighten_webcolor
from aiida_2d.visualize.core import (
    compute_projection,
    get_rotation_matrix,
    initialise_element_groups,
    VESTA_ELEMENT_INFO,
)
from aiida_2d.visualize.gui import AtomGui, AtomImages
from aiida_2d.visualize.svg import (
    create_axes_elements,
    create_svg_document,
    generate_svg_elements,
)
from aiida_2d.visualize.three import create_world_axes, generate_3js_render


def is_positive_number(self, attribute, value):
    """Validate that an attribute value is a number >= 0."""
    if not isinstance(value, (int, float)) or value < 0:
        raise TypeError(
            f"'{attribute.name}' must be a positive number (got {value!r} that is a "
            f"{value.__class__!r}).",
            attribute,
            (int, float),
            value,
        )


def is_html_color(self, attribute, value):
    """Validate that an attribute value is a valid HTML color."""
    try:
        if not isinstance(value, str):
            raise TypeError
        Color(value)
    except Exception:
        raise TypeError(
            f"'{attribute.name}' must be a valid hex color (got {value!r} that is a "
            f"{value.__class__!r}).",
            attribute,
            str,
            value,
        )


MILLER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["index"],
        "properties": {
            "index": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "number"},
            },
            "as_poly": {"type": "boolean"},
            "stroke_width": {"type": "number", "minimum": 0},
            "color": {"type": "string"},
        },
    },
}


@attr.s(kw_only=True)
class ViewConfig:
    """Configuration settings for atom visualisations."""

    rotations: str = attr.ib(default="", validator=instance_of(str))
    # string format of unit cell rotations '50x,-10y,120z' (note: order matters)
    element_colors: str = attr.ib(default="ase", validator=in_(("ase", "vesta")))
    element_radii: str = attr.ib(default="ase", validator=in_(("ase", "vesta")))
    radii_scale: float = attr.ib(default=0.89, validator=is_positive_number)
    atom_show_label: bool = attr.ib(default=True, validator=instance_of(bool))
    atom_label_by: str = attr.ib(
        default="element",
        validator=in_(("element", "index", "tag", "magmom", "charge")),
    )
    atom_font_size: int = attr.ib(default=14, validator=is_positive_number)
    atom_font_color: int = attr.ib(default="black", validator=is_html_color)
    atom_stroke_width: float = attr.ib(default=1.0, validator=is_positive_number)
    atom_color_by: str = attr.ib(default="element", validator=instance_of(str))
    atom_colormap: str = attr.ib(default="jet", validator=instance_of(str))
    atom_colormap_range: Union[list, tuple] = attr.ib(
        default=(None, None), validator=instance_of((list, tuple))
    )
    atom_lighten_by_depth: float = attr.ib(default=0.0, validator=is_positive_number)
    # Fraction (0 to 1) by which to lighten atom colors,
    # based on their fractional distance along the line from the
    # maximum to minimum z-coordinate of all elements
    atom_opacity: float = attr.ib(default=0.95, validator=is_positive_number)
    force_vector_scale: float = attr.ib(default=1.0, validator=is_positive_number)
    velocity_vector_scale: float = attr.ib(default=1.0, validator=is_positive_number)
    ghost_stroke_width: float = attr.ib(default=0.0, validator=is_positive_number)
    ghost_lighten: float = attr.ib(default=0.0, validator=is_positive_number)
    ghost_opacity: float = attr.ib(default=0.4, validator=is_positive_number)
    ghost_show_label: bool = attr.ib(default=False, validator=instance_of(bool))
    ghost_cross_out: bool = attr.ib(default=False, validator=instance_of(bool))
    show_unit_cell: bool = attr.ib(default=True, validator=instance_of(bool))
    show_uc_repeats: Union[bool, list] = attr.ib(
        default=False, validator=instance_of((bool, list, tuple))
    )
    uc_dash_pattern: Union[None, tuple] = attr.ib(default=None)
    uc_color: str = attr.ib(default="black", validator=is_html_color)
    show_bonds: bool = attr.ib(default=False, validator=instance_of(bool))
    bond_opacity: float = attr.ib(default=0.8, validator=is_positive_number)
    show_miller_planes: bool = attr.ib(default=True, validator=instance_of(bool))
    miller_planes: list = attr.ib(
        default=attr.Factory(list), validator=instance_of(list)
    )
    miller_fill_opacity: float = attr.ib(default=0.5, validator=is_positive_number)
    miller_stroke_opacity: float = attr.ib(default=0.9, validator=is_positive_number)
    show_axes: bool = attr.ib(default=True, validator=instance_of(bool))
    axes_length: float = attr.ib(default=15, validator=is_positive_number)
    axes_font_size: int = attr.ib(default=14, validator=is_positive_number)
    axes_line_color: str = attr.ib(default="black", validator=is_html_color)
    canvas_size: Union[list, tuple] = attr.ib(
        default=(400, 400), validator=instance_of((list, tuple))
    )
    canvas_color_foreground: str = attr.ib(default="#000000", validator=is_html_color)
    canvas_color_background: str = attr.ib(default="#ffffff", validator=is_html_color)
    zoom: float = attr.ib(default=1.0, validator=is_positive_number)
    canvas_crop: Union[list, tuple, None] = attr.ib(default=None)
    gui_swap_mouse: bool = attr.ib(default=False, validator=instance_of(bool))

    @uc_dash_pattern.validator
    def _validate_uc_dash_pattern(self, attribute, value):
        """Validate uc_dash_pattern is of form (solid, gap)."""
        if value is None:
            return
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise TypeError(
                f"'{attribute.name}' must be of the form (line_length, gap_length)."
            )
        if value[0] <= 0 or value[1] <= 0:
            raise TypeError(
                f"'{attribute.name}' (line_length, gap_length) must have positive lengths."
            )

    @canvas_crop.validator
    def _validate_canvas_crop(self, attribute, value):
        """Validate crop is of form (left, right, top, bottom)."""
        if value is None:
            return
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise TypeError(
                f"'{attribute.name}' must be of the form (left, right, top, bottom)."
            )

    @miller_planes.validator
    def _validate_miller_planes(self, attribute, value):
        """Validate miller_planes."""
        try:
            jsonschema.validate(value, MILLER_SCHEMA)
        except jsonschema.ValidationError as err:
            raise TypeError(f"'{attribute.name}' failed validation.") from err
        for plane in value:
            if "color" in plane:
                is_html_color(self, attribute, plane["color"])

    def __setattr__(self, key, value):
        """Add attr validation when setting attributes."""
        x_attr = getattr(attr.fields(self.__class__), key)
        if x_attr.validator:
            x_attr.validator(self, x_attr, value)
        super().__setattr__(key, value)


class AseView:
    """Class for visualising ``ase.Atoms`` or ``pymatgen.Structure``."""

    def __init__(self, config=None, **kwargs):
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
        """Return the configuration as a dictionary."""
        return attr.asdict(self._config)

    def add_miller_plane(
        self, h, k, l, *, as_poly=False, color="blue", stroke_width=1, reset=False
    ):
        """Add a miller plane to the config.

        Parameters
        ----------
        h : int
        k : int
        l : int
        as_poly : bool
            Render the plane a polygon, otherwise as lines
        color : str or tuple
            color of plane
        stroke_width : int
            width of outline
        reset : bool
            if True, remove any previously set miller planes

        """
        plane = [
            {
                "index": [h, k, l],
                "color": color,
                "stroke_width": stroke_width,
                "as_poly": as_poly,
            }
        ]
        if reset:
            self.config.miller_planes = plane
        else:
            self.config.miller_planes = self.config.miller_planes + plane

    def get_element_colors(self):
        """Return mapping of element atomic number to (hex) color."""
        if self.config.element_colors == "ase":
            return [
                "#{0:02X}{1:02X}{2:02X}".format(*(int(x * 255) for x in c))
                for c in ase_element_colors
            ]
        if self.config.element_colors == "vesta":
            return [
                "#{0:02X}{1:02X}{2:02X}".format(*(int(x * 255) for x in d[4:]))
                for d in VESTA_ELEMENT_INFO
            ]
        raise ValueError(self.config.element_colors)

    def get_element_radii(self):
        """Return mapping of element atomic number to atom radii."""
        if self.config.element_radii == "ase":
            return ase_covalent_radii.copy()
        if self.config.element_radii == "vesta":
            return [d[1] for d in VESTA_ELEMENT_INFO]
        raise ValueError(self.config.element_radii)

    def get_atom_colors(self, atoms):
        """Return mapping of atom index to (hex) color."""
        if self.config.atom_color_by == "element":
            element_colors = self.get_element_colors()
            return [element_colors[z] for z in atoms.numbers]

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
        else:
            # TODO can color by neighbours
            # TODO could also color by array atoms.get_array(self.config.atom_color_by)
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
        if self.config.atom_label_by == "element":
            if "occupancy" in atoms.info:
                return [
                    ",".join(atoms.info["occupancy"][t].keys())
                    for t in atoms.get_tags()
                ]
            return atoms.get_chemical_symbols()
        if self.config.atom_label_by == "index":
            return list(range(len(atoms)))
        if self.config.atom_label_by == "tag":
            return atoms.get_tags()
        if self.config.atom_label_by == "magmom":
            return atoms.get_initial_magnetic_moments()
        if self.config.atom_label_by == "charge":
            return atoms.get_initial_charges()
        # TODO could also label by array atoms.get_array(self.config.atom_label_by)
        raise ValueError(self.config.atom_label_by)

    def _convert_atoms(self, atoms, to_structure=False):
        """Convert ``pymatgen.Structure`` to/from ``ase.Atoms``.

        We preserve site properties, by storing them as arrays.
        """
        if isinstance(atoms, ase.Atoms) and not to_structure:
            return atoms

        from pymatgen import Structure
        from pymatgen.io.ase import AseAtomsAdaptor

        if isinstance(atoms, ase.Atoms) and to_structure:
            return AseAtomsAdaptor.get_structure(atoms)

        if isinstance(atoms, Structure) and not to_structure:
            structure = atoms
            atoms = AseAtomsAdaptor.get_atoms(atoms)
            for key, array in structure.site_properties.items():
                if key not in atoms.arrays:
                    atoms.set_array(key, np.array(array))
            # TODO propogate partial occupancies
            return atoms

        if isinstance(atoms, Structure) and to_structure:
            return atoms

        raise TypeError(f"atoms class not recognised: {atoms.__class__}")

    def _initialise_elements(self, atoms, center_in_uc=False, repeat_uc=(1, 1, 1)):
        """Prepare visualisation elements, in a backend agnostic manner."""
        config = self._config

        atoms = self._convert_atoms(atoms)

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
                    "color": [
                        config.miller_planes[i].get("color", "blue")
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
                },
                element=True,
            )
            element_groups[miller_type].set_property_many(
                {
                    "fill_opacity": config.miller_fill_opacity,
                    "stroke_opacity": config.miller_stroke_opacity,
                },
                element=False,
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
        structure = self._convert_atoms(atoms, to_structure=True)
        struct_data = structure.as_dict()
        config_data = self.get_config_as_dict()
        data_str = json.dumps(
            {
                "structure": struct_data,
                "config": config_data,
                "kwargs": {"center_in_uc": center_in_uc, "repeat_uc": repeat_uc},
            }
        )
        process = subprocess.Popen(
            "aiida-2d.view_atoms", stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.stdin.write(data_str.encode())
        process.stdin.close()
        sleep(test_init)
        if process.poll():
            raise RuntimeError(process.stderr.read().decode())
        return process

    def make_render(self, atoms, center_in_uc=False, repeat_uc=(1, 1, 1)):
        """Create a pythreejs render of the atoms or structure."""
        config = self.config
        atoms, element_groups = self._initialise_elements(
            atoms, center_in_uc=center_in_uc, repeat_uc=repeat_uc
        )
        rotation_matrix = get_rotation_matrix(config.rotations) * (-1, 1, -1)
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

        renderer = generate_3js_render(
            element_groups, canvas_size=config.canvas_size, zoom=config.zoom
        )
        if config.show_axes:
            import ipywidgets

            ax_renderer = create_world_axes(
                renderer.camera, renderer.controls[0], initial_rotation=rotation_matrix
            )
            renderer = ipywidgets.HBox([renderer, ax_renderer])

        return renderer


AseView.__init__.__doc__ = (
    "kwargs are used to initialise SVGConfig:"
    f"\n{inspect.signature(ViewConfig.__init__)}"
)


def _launch_gui_exec(json_string=None):
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
    structure_dict = data.pop("structure", {})
    config_dict = data.pop("config", {})
    kwargs = data.pop("kwargs", {})

    module = importlib.import_module(structure_dict["@module"])
    klass = getattr(module, structure_dict["@class"])
    structure = klass.from_dict(
        {k: v for k, v in structure_dict.items() if not k.startswith("@")}
    )
    ase_view = AseView(**config_dict)
    ase_view.make_gui(structure, **kwargs)


# Note: original commands (when creating SVG via tkinter postscript)
# gui.window.win.withdraw()  # hide window
# canvas = gui.window.canvas
# canvas.config(width=100, height=100); gui.draw()  # resize canvas
# gui.scale *= zoom; gui.draw()  # zoom
# canvas.postscript(file=fname)  # save canvas
