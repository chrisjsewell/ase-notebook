"""A module for creating an SVG visualisation of a structure."""
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

from aiida_2d.visualize.color import Color
from aiida_2d.visualize.core import (
    compute_projection,
    initialise_element_groups,
    rotate,
    VESTA_ELEMENT_INFO,
)
from aiida_2d.visualize.gui import AtomGui, AtomImages
from aiida_2d.visualize.svg import create_axes_elements, Drawing, generate_svg_elements


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

    # repeat: Union[list, tuple] = attr.ib(
    #     default=(1, 1, 1), validator=instance_of((list, tuple))
    # )
    # center: bool = attr.ib(default=False, validator=instance_of(bool))
    rotations: str = attr.ib(default="", validator=instance_of(str))
    element_colors: str = attr.ib(default="ase", validator=in_(("ase", "vesta")))
    element_radii: str = attr.ib(default="ase", validator=in_(("ase", "vesta")))
    radii_scale: float = attr.ib(default=0.89, validator=is_positive_number)
    atom_show_label: bool = attr.ib(default=True, validator=instance_of(bool))
    atom_label_by: str = attr.ib(
        default="element",
        validator=in_(("element", "index", "tag", "magmom", "charge")),
    )
    atom_font_size: int = attr.ib(default=14, validator=is_positive_number)
    atom_stroke_width: float = attr.ib(default=1.0, validator=is_positive_number)
    atom_color_by: str = attr.ib(default="element", validator=instance_of(str))
    atom_colormap: str = attr.ib(default="jet", validator=instance_of(str))
    atom_colormap_range: Union[list, tuple] = attr.ib(
        default=(None, None), validator=instance_of((list, tuple))
    )
    atom_opacity: float = attr.ib(default=0.95, validator=is_positive_number)
    ghost_stroke_width: float = attr.ib(default=0.0, validator=is_positive_number)
    ghost_lighten: float = attr.ib(default=0.0, validator=is_positive_number)
    ghost_opacity: float = attr.ib(default=0.4, validator=is_positive_number)
    ghost_show_label: bool = attr.ib(default=False, validator=instance_of(bool))
    show_uc: bool = attr.ib(default=True, validator=instance_of(bool))
    show_uc_repeats: Union[bool, list] = attr.ib(
        default=False, validator=instance_of((bool, list, tuple))
    )
    uc_segments: float = attr.ib(default=None)
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
    canvas_size: Union[list, tuple] = attr.ib(
        default=(400, 400), validator=instance_of((list, tuple))
    )
    canvas_color_foreground: str = attr.ib(default="#000000", validator=is_html_color)
    canvas_color_background: str = attr.ib(default="#ffffff", validator=is_html_color)
    zoom: float = attr.ib(default=1.0, validator=is_positive_number)
    crop_fraction: Union[list, tuple] = attr.ib(
        default=(1.0, 1.0), validator=instance_of((list, tuple))
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
            return atoms

        if isinstance(atoms, Structure) and to_structure:
            return atoms

        raise TypeError(f"atoms class not recognised: {atoms.__class__}")

    def _prepare_elements(self, atoms, center_in_uc=False, repeat_uc=(1, 1, 1)):
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

        atom_radii = self.get_atom_radii(atoms)
        atom_colors = self.get_atom_colors(atoms)
        # TODO lighten color by z-depth, or if ghost
        atom_labels = self.get_atom_labels(atoms)
        ghost_atoms = (
            atoms.get_array("ghost")
            if "ghost" in atoms.arrays
            else [False for _ in atoms]
        )

        element_groups = initialise_element_groups(
            atoms,
            atom_radii,
            show_uc=config.show_uc,
            uc_segments=config.uc_segments,
            show_bonds=config.show_bonds,
            miller_planes=config.miller_planes if config.show_miller_planes else None,
        )

        rotation_matrix = rotate(config.rotations)

        center, scale = compute_projection(
            element_groups, config.canvas_size, rotation_matrix
        )
        scale *= config.zoom

        axes = scale * rotation_matrix * (1, -1, 1)
        offset = np.dot(center, axes)
        offset[:2] -= 0.5 * np.array(config.canvas_size)
        element_groups.update_positions(axes, offset, scale)
        if config.show_bonds:
            element_groups["atoms"]._scale *= 0.65  # TODO accessing protected attribute

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
                "stroke_width": [
                    config.ghost_stroke_width if g else config.atom_stroke_width
                    for g in ghost_atoms
                ],
            },
            element=True,
        )
        element_groups["atoms"].set_property_many(
            {"font_size": config.atom_font_size}, element=False
        )
        element_groups["bond_lines"].set_property(
            "color",
            [
                (atom_colors[e.atom_index[0]], atom_colors[e.atom_index[1]])
                for e in element_groups["bond_lines"]
            ],
            element=True,
        )
        element_groups["bond_lines"].set_property_many(
            {"stroke_width": scale * 0.15, "stroke_opacity": config.bond_opacity},
            element=False,
        )
        for miller_type in ["miller_lines", "miller_planes"]:
            element_groups[miller_type].set_property_many(
                {
                    "color": [
                        config.miller_planes[el.index].get("color", "blue")
                        for el in element_groups[miller_type]
                    ],
                    "stroke_width": [
                        config.miller_planes[el.index].get("stroke_width", 1)
                        for el in element_groups[miller_type]
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

        return atoms, element_groups, rotation_matrix, scale

    def make_svg(self, atoms, center_in_uc=False, repeat_uc=(1, 1, 1)):
        """Create an SVG of the atoms or structure."""
        config = self.config
        atoms, element_groups, rotation_matrix, scale = self._prepare_elements(
            atoms, center_in_uc=center_in_uc, repeat_uc=repeat_uc
        )
        # TODO lighten ghost atoms, if config.ghost_lighten
        svg_elements = generate_svg_elements(element_groups, scale)
        # TODO cropping should be from all sides
        canvas_size = (np.array(config.canvas_size) * config.crop_fraction).tolist()
        if config.show_axes:
            svg_elements.extend(
                create_axes_elements(
                    rotation_matrix * (1, -1, 1),
                    canvas_size,
                    length=config.axes_length,
                    font_size=config.axes_font_size,
                )
            )
        # TODO use config.canvas_color_foreground and config.canvas_color_background
        dwg = Drawing("ase.svg", profile="tiny", size=canvas_size)
        for svg_element in svg_elements:
            dwg.add(svg_element)
        return dwg

    def make_gui(
        self,
        atoms,
        center_in_uc=False,
        repeat_uc=(1, 1, 1),
        bring_to_top=True,
        launch=True,
    ):
        """Launch a (blocking) GUI to view the atoms or structure."""
        config = self.config

        atoms, element_groups, rotation_matrix, scale = self._prepare_elements(
            atoms, center_in_uc=center_in_uc, repeat_uc=repeat_uc
        )

        images = AtomImages(
            [atoms],
            element_radii=np.array(self.get_element_radii()).tolist(),
            radii_scale=config.radii_scale,
        )

        label_sites = {"index": 1, "magmom": 2, "element": 3, "charge": 4}.get(
            config.atom_label_by, 0
        )
        if not config.atom_show_label:
            label_sites = 0
        config_settings = {
            "gui_foreground_color": config.canvas_color_foreground,
            "gui_background_color": config.canvas_color_background,
            "radii_scale": config.radii_scale,
            "force_vector_scale": 1.0,
            "velocity_vector_scale": 1.0,
            "show_unit_cell": config.show_uc,
            "unit_cell_segmentation": config.uc_segments,
            "show_axes": config.show_axes,
            "show_bonds": config.show_bonds,
            "show_millers": config.show_miller_planes,
            "swap_mouse": False,
        }
        ghost_settings = {
            "display": True,
            "cross": False,
            "label": config.ghost_show_label,
            "lighten": config.ghost_lighten,
            "opacity": config.ghost_opacity,
            "linewidth": config.ghost_stroke_width,
        }

        gui = AtomGui(
            images,
            config_settings=config_settings,
            rotations=config.rotations,
            element_colors=self.get_element_colors(),
            label_sites=label_sites,
            ghost_settings=ghost_settings,
            miller_planes=config.miller_planes,
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


AseView.__init__.__doc__ = (
    "kwargs are used to initialise SVGConfig:"
    f"\n{inspect.signature(ViewConfig.__init__)}"
)


def _launch_gui_exec():
    """Launch a GUI, with json data parsed from stdin."""
    if sys.stdin.isatty():
        raise IOError("stdin is empty")
    data_str = sys.stdin.read()
    data = json.loads(data_str)
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


# gui.window.win.withdraw()  # hide window
# canvas = gui.window.canvas
# canvas.config(width=100, height=100); gui.draw()  # resize canvas
# gui.scale *= zoom; gui.draw()  # zoom
# canvas.postscript(file=fname)  # save canvas


# from svgutils.transform import fromstring
# image = fromstring(image)
