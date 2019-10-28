"""A module for creating an SVG visualisation of a structure."""
import inspect
from typing import Union

from ase.data import covalent_radii as ase_covalent_radii
from ase.data.colors import jmol_colors as ase_element_colors
import attr
from attr.validators import in_, instance_of
import jsonschema
import numpy as np
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from svgwrite import Drawing, shapes, text

from aiida_2d.visualize.core import (
    initialise_element_groups,
    rotate,
    VESTA_ELEMENT_INFO,
)


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


def generate_svg_elements(element_group, scale, atom_radii):
    """Create the SVG elements, related to the 3D objects."""
    svg_elements = []

    for element in element_group.yield_zorder():
        if element.name == "atoms":
            if not element.get("visible", True):
                continue
            svg_elements.append(
                shapes.Circle(
                    element.position[:2],
                    r=element.sradius,
                    fill=element.color,
                    fill_opacity=element.get("fill_opacity", 0.95),
                    stroke=element.get("stroke", "black"),
                    stroke_width=element.get("stroke_width", 1),
                )
            )
            if "label" in element and element.label is not None:
                svg_elements.append(
                    text.Text(
                        element.label,
                        x=(int(element.position[0]),),
                        y=(int(element.position[1]),),
                        text_anchor="middle",
                        dominant_baseline="middle",
                        font_size=element.get("font_size", 20),
                    )
                )
        if element.name == "cell_lines":
            svg_elements.append(
                shapes.Line(
                    element.position[0][:2],
                    element.position[1][:2],
                    stroke="black",
                    stroke_dasharray=f"{element.get('dashed', '6,4')}",
                )
            )
        if element.name == "bond_lines":
            start, end = element.position[0][:2], element.position[1][:2]
            svg_elements.append(
                shapes.Line(
                    start,
                    start + 0.5 * (end - start),
                    stroke=element.color[0],
                    stroke_width=element.get("stroke_width", 1),
                    stroke_linecap="round",
                    stroke_opacity=element.get("stroke_opacity", 0.8),
                )
            )
            svg_elements.append(
                shapes.Line(
                    start + 0.5 * (end - start),
                    end,
                    stroke=element.color[1],
                    stroke_width=element.get("stroke_width", 1),
                    stroke_linecap="round",
                    stroke_opacity=element.get("stroke_opacity", 0.8),
                )
            )
        if element.name == "miller_lines":
            svg_elements.append(
                shapes.Line(
                    element.position[0][:2],
                    element.position[1][:2],
                    stroke=element.get("color", "blue"),
                    stroke_width=element.get("stroke_width", 1),
                    stroke_opacity=element.get("stroke_opacity", 0.8),
                )
            )
        if element.name == "miller_planes":
            svg_elements.append(
                shapes.Polygon(
                    points=element.position[:, :2],
                    fill=element.get("color", "blue"),
                    fill_opacity=element.get("fill_opacity", 0.5),
                    stroke=element.get("color", "blue"),
                    stroke_width=element.get("stroke_width", 0),
                    stroke_opacity=element.get("stroke_opacity", 0.5),
                )
            )
    return svg_elements


def create_axes_elements(
    axes, window_size, length=15, width=1, font_size=14, offset=20, font_offset=1.0
):
    """Create the SVG elements, related to the axes."""
    rgb = ["red", "green", "blue"]

    svg_elements = []

    for i in axes[:, 2].argsort():
        a = offset
        b = window_size[1] - offset
        c = int(axes[i][0] * length + a)
        d = int(axes[i][1] * length + b)
        e = int(axes[i][0] * length * font_offset + a)
        f = int(axes[i][1] * length * font_offset + b)
        svg_elements.append(
            shapes.Line([a, b], [c, d], stroke="black", stroke_width=width)
        )
        svg_elements.append(
            text.Text(
                "XYZ"[i],
                x=(e,),
                y=(f,),
                fill=rgb[i],
                text_anchor="middle",
                dominant_baseline="middle",
                font_size=font_size,
            )
        )

    return svg_elements


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
            "color": {"type": ["string", "array"]},
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

    def __setattr__(self, key, value):
        """Add attr validation when setting attributes."""
        x_attr = getattr(attr.fields(self.__class__), key)
        if x_attr.validator:
            x_attr.validator(self, x_attr, value)
        super().__setattr__(key, value)


class AseView:
    """Class for visualising ``ase.Atoms`` or ``pymatgen.Structure``."""

    def __init__(self, **kwargs):
        """This is replaced by ``SVGConfig`` docstring."""  # noqa: D401
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

    def make_svg(self, atoms, center_in_uc=False, repeat_uc=(1, 1, 1)):
        """Create an SVG of the atoms or structure."""
        config = self._config

        if isinstance(atoms, Structure):
            structure = atoms
            atoms = AseAtomsAdaptor.get_atoms(atoms)
            # preserve site properties, by storing them as arrays
            for key, array in structure.site_properties.items():
                if key not in atoms.arrays:
                    atoms.set_array(key, np.array(array))

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

        svg_elements = generate_svg_elements(element_groups, scale, atom_radii)
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
        dwg = Drawing("ase.svg", profile="tiny", size=canvas_size)
        for svg_element in svg_elements:
            dwg.add(svg_element)
        return dwg


AseView.__init__.__doc__ = (
    "kwargs are used to initialise SVGConfig:"
    f"\n{inspect.signature(ViewConfig.__init__)}"
)
