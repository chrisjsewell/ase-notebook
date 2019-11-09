"""A module for creating visualisations of a structure."""
from collections import Mapping
from typing import Tuple, Union

import attr
from attr.validators import in_, instance_of

from .color import Color


def in_range(minimum=None, maximum=None):
    """Validate that an attribute value is a number >= 0."""  # noqa: D202

    def _validate(self, attribute, value):
        raise_error = False
        if not isinstance(value, (int, float)):
            raise_error = True
        elif minimum is not None and value < minimum:
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


@attr.s(kw_only=True)
class ViewConfig:
    """Configuration settings for initialisation of atom visualisations."""

    rotations: str = attr.ib(default="", validator=instance_of(str))
    # string format of unit cell rotations '50x,-10y,120z' (note: order matters)
    element_colors: str = attr.ib(default="ase", validator=in_(("ase", "vesta")))
    element_radii: str = attr.ib(default="ase", validator=in_(("ase", "vesta")))
    radii_scale: float = attr.ib(default=0.89, validator=in_range(0))
    atom_show_label: bool = attr.ib(default=True, validator=instance_of(bool))
    atom_label_by: str = attr.ib(
        default="element",
        validator=in_(("element", "index", "tag", "magmom", "charge", "array")),
    )
    atom_label_array: str = attr.ib(default="", validator=instance_of(str))
    atom_font_size: int = attr.ib(default=14, validator=[instance_of(int), in_range(1)])
    atom_font_color: str = attr.ib(default="black", validator=is_html_color)
    atom_stroke_width: float = attr.ib(default=1.0, validator=in_range(0))
    atom_stroke_opacity: float = attr.ib(default=0.95, validator=in_range(0, 1))
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
    )
    atom_color_array: str = attr.ib(default="", validator=instance_of(str))
    atom_colormap: str = attr.ib(default="jet", validator=instance_of(str))
    atom_colormap_range: Union[list, tuple] = attr.ib(
        default=(None, None), validator=instance_of((list, tuple))
    )
    atom_lighten_by_depth: float = attr.ib(default=0.0, validator=in_range(0))
    # Fraction (0 to 1) by which to lighten atom colors,
    # based on their fractional distance along the line from the
    # maximum to minimum z-coordinate of all elements
    atom_opacity: float = attr.ib(default=0.95, validator=in_range(0, 1))
    force_vector_scale: float = attr.ib(default=1.0, validator=in_range(0))
    velocity_vector_scale: float = attr.ib(default=1.0, validator=in_range(0))
    ghost_stroke_width: float = attr.ib(default=0.0, validator=in_range(0))
    ghost_lighten: float = attr.ib(default=0.0, validator=in_range(0))
    ghost_opacity: float = attr.ib(default=0.4, validator=in_range(0, 1))
    ghost_stroke_opacity: float = attr.ib(default=0.4, validator=in_range(0, 1))
    ghost_show_label: bool = attr.ib(default=False, validator=instance_of(bool))
    ghost_cross_out: bool = attr.ib(default=False, validator=instance_of(bool))
    show_unit_cell: bool = attr.ib(default=True, validator=instance_of(bool))
    show_uc_repeats: Union[bool, list] = attr.ib(
        default=False, validator=instance_of((bool, list, tuple))
    )
    uc_dash_pattern: Union[None, tuple] = attr.ib(default=None)
    uc_color: str = attr.ib(default="black", validator=is_html_color)
    show_bonds: bool = attr.ib(default=False, validator=instance_of(bool))
    bond_opacity: float = attr.ib(default=0.8, validator=in_range(0, 1))
    show_miller_planes: bool = attr.ib(default=True, validator=instance_of(bool))
    miller_planes: Tuple[dict] = attr.ib(
        default=(), converter=convert_to_miller_dicts, validator=instance_of(tuple)
    )
    miller_as_lines: bool = attr.ib(default=False, validator=instance_of(bool))
    show_axes: bool = attr.ib(default=True, validator=instance_of(bool))
    axes_length: float = attr.ib(default=15, validator=in_range(0))
    axes_font_size: int = attr.ib(default=14, validator=[instance_of(int), in_range(1)])
    axes_line_color: str = attr.ib(default="black", validator=is_html_color)
    canvas_size: Tuple[float, float] = attr.ib(
        default=(400, 400), validator=instance_of((list, tuple))
    )
    canvas_color_foreground: str = attr.ib(default="#000000", validator=is_html_color)
    canvas_color_background: str = attr.ib(default="#ffffff", validator=is_html_color)
    canvas_background_opacity: float = attr.ib(default=0.0, validator=in_range(0, 1))
    canvas_crop: Union[list, tuple, None] = attr.ib(default=None)
    zoom: float = attr.ib(default=1.0, validator=in_range(0))
    camera_fov: float = attr.ib(default=10.0, validator=in_range(1))
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

    def __setattr__(self, key, value):
        """Add attr conversion and validation when setting attributes."""
        x_attr = getattr(attr.fields(self.__class__), key)  # type: attr.Attribute
        if x_attr.converter:
            value = x_attr.converter(value)
        if x_attr.validator:
            x_attr.validator(self, x_attr, value)
        super().__setattr__(key, value)
