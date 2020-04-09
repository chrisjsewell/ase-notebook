"""A module for creating visualisations of a structure."""
from collections import Iterable, Mapping
from typing import Optional, Tuple, Union

import attr
from attr.validators import deep_iterable, in_, instance_of, optional

from .attr_doc import autodoc
from .color import Color


def iterable_length(length):
    """Validate that an attribute is an iterable of a certain length."""  # noqa: D202

    def _validate(self, attribute, value):
        if not isinstance(value, Iterable) or len(value) != 2:
            raise TypeError(
                f"'{attribute.name}' must be an iterable of length {length} "
                f"(got {value!r} that is a {value.__class__!r}).",
                attribute,
                Iterable,
                value,
            )

    return _validate


def in_range(minimum=None, maximum=None, inclusive_min=True):
    """Validate that an attribute value is a number >= 0."""  # noqa: D202

    def _validate(self, attribute, value):
        raise_error = False
        if not isinstance(value, (int, float)):
            raise_error = True
        elif minimum is not None and (
            (inclusive_min and value < minimum)
            or (not inclusive_min and value <= minimum)
        ):
            raise_error = True
        elif maximum is not None and value > maximum:
            raise_error = True
        if raise_error:
            raise TypeError(
                f"'{attribute.name}' must be a number in range [{minimum},{maximum}] "
                f"(got {value!r} that is a {value.__class__!r}).",
                attribute,
                (int, float),
                value,
            )

    return _validate


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


@autodoc
@attr.s(slots=True)
class MillerPlane:
    """A data class to define a Miller Plane to visualise."""

    h: float = attr.ib(default=0, validator=instance_of((float, int)))
    k: float = attr.ib(default=0, validator=instance_of((float, int)))
    l: float = attr.ib(default=0, validator=instance_of((float, int)))
    fill_color: str = attr.ib(default="red", validator=is_html_color)
    stroke_color: str = attr.ib(default="red", validator=is_html_color)
    stroke_width: float = attr.ib(default=1.0, validator=in_range(0))
    fill_opacity: float = attr.ib(default=0.5, validator=in_range(0, 1))
    stroke_opacity: float = attr.ib(default=0.9, validator=in_range(0, 1))

    @h.validator
    def _validate_h(self, attribute, value):
        if not any([self.h, self.k, self.l]):
            raise ValueError("at least one of h, k, l must be non-zero")

    @k.validator
    def _validate_k(self, attribute, value):
        if not any([self.h, self.k, self.l]):
            raise ValueError("at least one of h, k, l must be non-zero")

    @l.validator
    def _validate_l(self, attribute, value):
        if not any([self.h, self.k, self.l]):
            raise ValueError("at least one of h, k, l must be non-zero")


def convert_to_miller_dicts(iterable):
    """Convert items in a list to MillerPlane validated dictionaries."""
    output = []
    for miller in iterable:
        if isinstance(miller, MillerPlane):
            pass
        elif isinstance(miller, Mapping):
            miller = MillerPlane(**miller)
        else:
            miller = MillerPlane(*miller)
        output.append(attr.asdict(miller))
    return tuple(output)


@autodoc
@attr.s(kw_only=True)
class ViewConfig:
    """Configuration settings for initialisation of atom visualisations."""

    rotations: str = attr.ib(
        default="",
        validator=instance_of(str),
        metadata={
            "help": (
                "Initial unit cell rotation in string format, "
                "e.g. '50x,-10y,120z' (note: order matters)"
            )
        },
    )
    element_colors: str = attr.ib(
        default="ase",
        validator=in_(("ase", "vesta")),
        metadata={
            "help": "Element to color mapping to use, if ``atom_color_by='element'``."
        },
    )
    element_radii: str = attr.ib(
        default="ase",
        validator=in_(("ase", "vesta")),
        metadata={"help": "Element to color mapping to use."},
    )
    radii_scale: float = attr.ib(
        default=0.89,
        validator=in_range(0),
        metadata={"help": "Scale all radii by this value."},
    )
    atom_show_label: bool = attr.ib(
        default=True,
        validator=instance_of(bool),
        metadata={"help": "Add atom labels to visualisation."},
    )
    atom_label_by: str = attr.ib(
        default="element",
        validator=in_(("element", "index", "tag", "magmom", "charge", "array")),
        metadata={"help": "Atom property to label atoms by."},
    )
    atom_label_array: str = attr.ib(
        default="",
        validator=instance_of(str),
        metadata={
            "help": "The array name to use, if ``array`` chosen for ``atom_label_by``"
        },
    )
    atom_font_size: int = attr.ib(
        default=14,
        validator=[instance_of(int), in_range(1)],
        metadata={"help": "Font size for atom labels."},
    )
    atom_font_color: str = attr.ib(
        default="black",
        validator=is_html_color,
        metadata={"help": "Font size for atom labels."},
    )
    atom_stroke_width: float = attr.ib(
        default=1.0,
        validator=in_range(0),
        metadata={"help": "line width for atom outlines."},
    )
    atom_stroke_opacity: float = attr.ib(
        default=0.95,
        validator=in_range(0, 1),
        metadata={"help": "line opacity for atom outlines."},
    )
    atom_color_by: str = attr.ib(
        default="element",
        validator=in_(
            (
                "element",
                "index",
                "tag",
                "magmom",
                "charge",
                "velocity",
                "color_array",
                "value_array",
            )
        ),
        metadata={"help": "Atom property to color atoms by."},
    )
    atom_color_array: str = attr.ib(
        default="",
        validator=instance_of(str),
        metadata={
            "help": "The array name to use, if ``array`` chosen for ``atom_color_by``"
        },
    )
    atom_colormap: str = attr.ib(
        default="jet",
        validator=instance_of(str),
        metadata={
            "help": "The matplotlib colormap to use with ``atom_color_by`` values"
        },
    )
    atom_colormap_range: Union[list, tuple] = attr.ib(
        default=(None, None),
        validator=instance_of((list, tuple)),
        metadata={
            "help": "The matplotlib colormap normalisation use with ``atom_color_by`` values"
        },
    )
    atom_lighten_by_depth: float = attr.ib(
        default=0.0,
        validator=in_range(0),
        metadata={
            "help": (
                "Fraction by which to lighten atom colors, "
                "based on their fractional distance along the line, "
                "from the maximum to minimum z-coordinate of all elements"
            )
        },
    )
    atom_opacity: float = attr.ib(
        default=0.95, validator=in_range(0, 1), metadata={"help": ""}
    )
    force_vector_scale: float = attr.ib(
        default=1.0,
        validator=in_range(0),
        metadata={"help": "Length of force vector arrows."},
    )
    velocity_vector_scale: float = attr.ib(
        default=1.0,
        validator=in_range(0),
        metadata={"help": "Length of velocity vector arrows."},
    )
    ghost_stroke_width: float = attr.ib(
        default=0.0, validator=in_range(0), metadata={"help": ""}
    )
    ghost_lighten: float = attr.ib(
        default=0.0,
        validator=in_range(0),
        metadata={"help": "Lighten the ghost atom colors by this fraction."},
    )
    ghost_opacity: float = attr.ib(
        default=0.4, validator=in_range(0, 1), metadata={"help": ""}
    )
    ghost_stroke_opacity: float = attr.ib(
        default=0.4, validator=in_range(0, 1), metadata={"help": ""}
    )
    ghost_show_label: bool = attr.ib(
        default=False, validator=instance_of(bool), metadata={"help": ""}
    )
    ghost_cross_out: bool = attr.ib(
        default=False, validator=instance_of(bool), metadata={"help": ""}
    )
    show_unit_cell: bool = attr.ib(
        default=True, validator=instance_of(bool), metadata={"help": ""}
    )
    show_uc_repeats: Union[bool, list] = attr.ib(
        default=False,
        validator=instance_of((bool, list, tuple)),
        metadata={
            "help": "If True, and the atoms have been repeated, show each unit cell of the original atoms."
        },
    )
    uc_dash_pattern: Union[None, Tuple[float, float]] = attr.ib(
        default=None,
        metadata={"help": "A (length, gap) dash pattern for unit cell lines."},
    )
    uc_color: str = attr.ib(
        default="black",
        validator=is_html_color,
        metadata={"help": "Unit cell line color."},
    )
    show_bonds: bool = attr.ib(
        default=False,
        validator=instance_of(bool),
        metadata={"help": "Show atomic bonds."},
    )
    bond_radii_scale: float = attr.ib(
        default=1.5,
        validator=in_range(0.0, inclusive_min=False),
        metadata={
            "help": "Factor to scale atomic radii by, when computing bonds (via overlapping radii)"
        },
    )
    bond_array_name: Optional[str] = attr.ib(
        default=None,
        validator=optional(instance_of(str)),
        metadata={
            "help": (
                "The name of a boolean array on the Atoms, "
                "specifying which atoms that bonds should be drawn for "
                "(if None, then all bonds are drawn)."
            )
        },
    )
    bond_pairs_filter: Optional[list] = attr.ib(
        default=None,
        metadata={
            "help": (
                "A list of bond element pairs to filter by, "
                "e.g. [('Fe', 'O'), ('Fe', 'Fe')]."
            )
        },
    )
    bond_opacity: float = attr.ib(
        default=0.8,
        validator=in_range(0, 1),
        metadata={"help": "Opacity of atomic bond lines."},
    )
    bond_color_by: str = attr.ib(
        default="atoms",
        validator=in_(("atoms", "length")),
        metadata={
            "help": "How to color bond: 'atoms' (same as connecting atoms) or 'length'."
        },
    )
    bond_colormap: str = attr.ib(
        default="jet",
        validator=instance_of(str),
        metadata={
            "help": ("The matplotlib colormap to use with ``bond_color_by='length'``")
        },
    )
    bond_colormap_range: Union[list, tuple] = attr.ib(
        default=(None, None),
        validator=deep_iterable(
            optional(instance_of((int, float))), iterable_length(2)
        ),
        metadata={
            "help": (
                "The matplotlib colormap normalisation use with "
                "``bond_color_by='length'``"
            )
        },
    )
    show_miller_planes: bool = attr.ib(
        default=True, validator=instance_of(bool), metadata={"help": ""}
    )
    miller_planes: Tuple[dict] = attr.ib(
        default=(),
        converter=convert_to_miller_dicts,
        validator=instance_of(tuple),
        metadata={
            "help": "List of dictionaries, describing miller index planes to create."
        },
    )
    miller_as_lines: bool = attr.ib(
        default=False,
        validator=instance_of(bool),
        metadata={
            "help": "If True, the miller planes will be created as lines, rather than polygons."
        },
    )
    show_axes: bool = attr.ib(
        default=True,
        validator=instance_of(bool),
        metadata={
            "help": ("Show the 'world' axes, " "at a corner of the visualisation.")
        },
    )
    axes_uc: bool = attr.ib(
        default=False,
        validator=instance_of(bool),
        metadata={
            "help": (
                "Show the 'unit cell' axes (abc), instead of the 'world' axes (XYZ)"
            )
        },
    )
    axes_length: float = attr.ib(
        default=15,
        validator=in_range(0, inclusive_min=False),
        metadata={"help": "Length of axes lines."},
    )
    axes_font_size: int = attr.ib(
        default=14, validator=[instance_of(int), in_range(1)], metadata={"help": ""}
    )
    axes_line_color: str = attr.ib(
        default="black", validator=is_html_color, metadata={"help": ""}
    )
    axes_offset: Tuple[float, float] = attr.ib(
        default=(20, 20),
        validator=deep_iterable(instance_of((int, float)), iterable_length(2)),
        metadata={"help": "Offset of axis origin from the bottom left of the canvas"},
    )
    canvas_size: Tuple[float, float] = attr.ib(
        default=(400, 400),
        validator=deep_iterable(in_range(1), iterable_length(2)),
        metadata={"help": ""},
    )
    canvas_color_foreground: str = attr.ib(
        default="#000000", validator=is_html_color, metadata={"help": ""}
    )
    canvas_color_background: str = attr.ib(
        default="#ffffff", validator=is_html_color, metadata={"help": ""}
    )
    canvas_background_opacity: float = attr.ib(
        default=0.0, validator=in_range(0, 1), metadata={"help": ""}
    )
    canvas_crop: Union[list, tuple, None] = attr.ib(
        default=None, metadata={"help": "Crop canvas: (left, right, top, bottom)"}
    )
    zoom: float = attr.ib(
        default=1.0,
        validator=in_range(0, inclusive_min=False),
        metadata={"help": "3D camera zoom."},
    )
    camera_fov: float = attr.ib(
        default=10.0,
        validator=in_range(1),
        metadata={"help": "3D camera field-of-view."},
    )
    gui_swap_mouse: bool = attr.ib(
        default=False,
        validator=instance_of(bool),
        metadata={"help": "Used with ``make_gui`` only."},
    )

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

    @bond_pairs_filter.validator
    def _bond_pairs_filter(self, attribute, value):
        """Validate `bond_pairs_filter` attribute."""
        if value is None:
            return
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"'{attribute.name}' must be a list or tuple.")
        for i, item in enumerate(value):
            try:
                assert (
                    len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], str)
                )
            except (AssertionError, IndexError, TypeError, ValueError):
                raise TypeError(
                    f"'{attribute.name}' item {i} must be a pair of strings."
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

    def __setattr__(self, key, value):
        """Add attr conversion and validation when setting attributes."""
        x_attr = getattr(attr.fields(self.__class__), key)  # type: attr.Attribute
        if x_attr.converter:
            value = x_attr.converter(value)
        if x_attr.validator:
            x_attr.validator(self, x_attr, value)
        super().__setattr__(key, value)
