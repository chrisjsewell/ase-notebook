"""A module for creating an SVG visualisation of a structure."""
import numpy as np
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from svgwrite import Drawing, shapes, text

from aiida_2d.visualize.core import (
    compute_element_coordinates,
    rotate,
    VESTA_ELEMENT_INFO,
)


def get_radii(atoms, element_radii, scale=0.89):
    """Get radii for each atom."""
    radii = np.array([element_radii[z] for z in atoms.numbers])
    radii *= scale
    return radii


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
            svg_elements.append(
                shapes.Circle(
                    element.position[:2],
                    r=element.sradius,
                    fill=element.color,
                    fill_opacity=element.get("fill_opacity", 0.95),
                    stroke="black",
                )
            )
            if "label" in element:
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
        if element.name == "miller_planes":
            svg_elements.append(
                shapes.Polygon(
                    points=element.position[:, :2],
                    fill=element.get("color", "blue"),
                    fill_opacity=element.get("fill_opacity", 0.5),
                    stroke=element.get("color", "blue"),
                    stroke_width=element.get("stroke_width", 0),
                    stroke_opacity=element.get("fill_opacity", 0.5),
                )
            )
    return svg_elements


def create_axes_elements(axes, window_size, length=15, width=1, font_size=14):
    """Create the SVG elements, related to the axes."""
    rgb = ["red", "green", "blue"]

    svg_elements = []

    for i in axes[:, 2].argsort():
        a = 20
        b = window_size[1] - 20
        c = int(axes[i][0] * length + a)
        d = int(axes[i][1] * length + b)
        svg_elements.append(
            shapes.Line([a, b], [c, d], stroke="black", stroke_width=width)
        )
        svg_elements.append(
            text.Text(
                "XYZ"[i],
                x=(int(c),),
                y=(int(d),),
                fill=rgb[i],
                text_anchor="middle",
                dominant_baseline="middle",
                font_size=font_size,
            )
        )

    return svg_elements


def create_svg(
    atoms,
    repeat=(1, 1, 1),
    center=False,
    rotations="",
    show_uc=True,
    show_unit_repeats=False,
    uc_segments=None,
    show_bonds=False,
    miller_planes=None,
    radii_scale=0.89,
    font_size=14,
    atom_opacity=0.95,
    bond_opacity=0.8,
    miller_opacity=0.5,
    size=(400, 400),
):
    """Create the SVG."""
    if isinstance(atoms, Structure):
        structure = atoms
        atoms = AseAtomsAdaptor.get_atoms(atoms)
        # preserve site properties, by storing them as arrays
        for key, array in structure.site_properties.items():
            if key not in atoms.arrays:
                atoms.set_array(key, np.array(array))

    if center:
        atoms.center()
    atoms = atoms.repeat(repeat)
    if show_unit_repeats:
        atoms.info["unit_cell_repeat"] = (
            repeat if isinstance(show_unit_repeats, bool) else show_unit_repeats
        )

    atom_radii = get_radii(atoms, [d[1] for d in VESTA_ELEMENT_INFO], scale=radii_scale)

    element_group = compute_element_coordinates(
        atoms,
        atom_radii,
        show_uc=show_uc,
        uc_segments=uc_segments,
        show_bonds=show_bonds,
        miller_planes=miller_planes,
    )

    rotation_matrix = rotate(rotations)

    center, scale = compute_projection(element_group, size, rotation_matrix)

    axes = scale * rotation_matrix * (1, -1, 1)
    offset = np.dot(center, axes)
    offset[:2] -= 0.5 * np.array(size)
    element_group.update_positions(axes, offset, scale)
    if show_bonds:
        element_group["atoms"]._scale *= 0.65  # TODO accessing protected attribute

    element_colors = [
        "#{0:02X}{1:02X}{2:02X}".format(*(int(x * 255) for x in d[4:]))
        for d in VESTA_ELEMENT_INFO
    ]
    element_group["atoms"].set_property_many(
        {
            "color": [element_colors[z] for z in atoms.numbers],
            "label": atoms.get_chemical_symbols(),
        },
        element=True,
    )
    element_group["atoms"].set_property_many(
        {"font_size": font_size, "fill_opacity": atom_opacity}, element=False
    )
    element_group["bond_lines"].set_property(
        "color",
        [
            (
                element_colors[atoms[e.atom_index[0]].number],
                element_colors[atoms[e.atom_index[1]].number],
            )
            for e in element_group["bond_lines"]
        ],
        element=True,
    )
    element_group["bond_lines"].set_property_many(
        {"stroke_width": scale * 0.15, "stroke_opacity": bond_opacity}, element=False
    )
    element_group["miller_planes"].set_property_many(
        {
            "color": [
                miller_planes[el.index][3] for el in element_group["miller_planes"]
            ],
            "stroke_width": [
                miller_planes[el.index][4] for el in element_group["miller_planes"]
            ],
        },
        element=True,
    )
    element_group["miller_planes"].set_property_many(
        {"fill_opacity": miller_opacity}, element=False
    )
    svg_elements = generate_svg_elements(element_group, scale, atom_radii)
    svg_elements.extend(create_axes_elements(rotation_matrix, size))
    dwg = Drawing("test.svg", profile="tiny", size=size)
    for svg_element in svg_elements:
        dwg.add(svg_element)
    return dwg
