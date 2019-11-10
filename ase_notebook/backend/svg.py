"""A module for creating an SVG visualisation of a structure."""
from itertools import cycle
import os
import tempfile

import numpy as np
from svgwrite import Drawing, path, shapes, text
from svgwrite.container import Group
from svgwrite.filters import Filter


def generate_svg_elements(element_group, element_colors=None, background_color="white"):
    """Create the SVG elements, related to the 3D objects.

    Parameters
    ----------
    element_group : ase_notebook.draw_elements.DrawGroup
        Container of all element groups to be created.
    background_color : str

    Returns
    -------
    list[svgwrite.base.BaseElement]

    """
    svg_elements = []

    for _, element in element_group.yield_zorder():
        if element.name == "atoms":
            if not element.get("visible", True):
                continue

            if element.occupancy is not None:
                from ase.data import atomic_numbers

                if (np.sum([o for o in element.occupancy.values()])) < 1.0:
                    # first draw an empty circle if a site is not fully occupied
                    svg_elements.append(
                        shapes.Circle(
                            element.position[:2],
                            r=element.sradius,
                            fill=background_color,
                            fill_opacity=element.get("fill_opacity", 0.95),
                            stroke=element.get("stroke", "black"),
                            stroke_width=element.get("stroke_width", 1),
                        )
                    )
                angle_start = 0
                # start with the dominant species
                for sym, occ in sorted(
                    element.occupancy.items(), key=lambda x: x[1], reverse=True
                ):
                    if np.round(occ, decimals=4) == 1.0:
                        svg_elements.append(
                            shapes.Circle(
                                element.position[:2],
                                r=element.sradius,
                                fill=element_colors[atomic_numbers[sym]],
                                fill_opacity=element.get("fill_opacity", 0.95),
                                stroke=element.get("stroke_color", "black"),
                                stroke_width=element.get("stroke_width", 1),
                            )
                        )
                    else:
                        angle_extent = 360.0 * occ
                        svg_elements.append(
                            create_arc_element(
                                element.position[:2],
                                angle_start,
                                angle_start + angle_extent,
                                element.sradius,
                                fill=element_colors[atomic_numbers[sym]],
                                fill_opacity=element.get("fill_opacity", 0.95),
                                stroke=element.get("stroke_color", "black"),
                                stroke_width=element.get("stroke_width", 1),
                            )
                        )
                        angle_start += angle_extent
            else:
                svg_elements.append(
                    shapes.Circle(
                        element.position[:2],
                        r=element.sradius,
                        fill=element.color,
                        fill_opacity=element.get("fill_opacity", 0.95),
                        stroke=element.get("stroke_color", "black"),
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
                        fill=element.get("font_color", "black"),
                    )
                )
            # TODO add force/velocity vectors
            # TODO add ghost crosses
        if element.name == "cell_lines":
            svg_elements.append(
                shapes.Line(
                    element.position[0][:2],
                    element.position[1][:2],
                    stroke=element.get("color", "black"),
                    # stroke_dasharray=f"{element.get('dashed', '6,4')}",
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
                    stroke=element.get("stroke_color", "blue"),
                    stroke_width=element.get("stroke_width", 1),
                    stroke_opacity=element.get("stroke_opacity", 0.8),
                )
            )
        if element.name == "miller_planes":
            svg_elements.append(
                shapes.Polygon(
                    points=element.position[:, :2],
                    fill=element.get("fill_color", "blue"),
                    fill_opacity=element.get("fill_opacity", 0.5),
                    stroke=element.get("stroke_color", "blue"),
                    stroke_width=element.get("stroke_width", 0),
                    stroke_opacity=element.get("stroke_opacity", 0.5),
                )
            )
    return svg_elements


def cart2polar(x, y):
    """Convert cartesian to polar coordinates."""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, np.rad2deg(phi))


def polar2cart(radius, angle):
    """Convert polar to cartesian coordinates."""
    x = radius * np.cos(np.radians(angle))
    y = radius * np.sin(np.radians(angle))
    return (x, y)


def create_arc_element(center, start, end, radius, **kwargs):
    """Create an arc (circle section) path element.

    Parameters
    ----------
    center: tuple
        (x, y)
    start: float
        starting angle from x axis (in degrees)
    end: float
        final angle from x axis (in degrees)
    radius: float

    Returns
    -------
    svgwrite.path.Path

    """
    c = np.array(center)
    p1 = np.array(polar2cart(radius, start)) + c
    p2 = np.array(polar2cart(radius, end)) + c

    l1 = p1 - c
    l2 = p2 - p1
    l3 = c - p2

    if start < end:
        angle_dir = 1
        large_arc = 1 if end - start >= 180 else 0
    else:
        angle_dir = 0
        large_arc = 1 if start - end >= 180 else 0

    return path.Path(
        [
            f"m{c[0]},{c[1]}",
            f"l{l1[0]},{l1[1]}",
            f"a{radius},{radius},{0},{large_arc},{angle_dir},{l2[0]},{l2[1]}",
            f"l{l3[0]},{l3[1]}",
        ],
        **kwargs,
    )


def create_axes_elements(
    axes,
    window_size,
    *,
    length=15,
    font_size=14,
    inset=(20, 20),
    font_offset=1.0,
    line_width=1,
    line_color="black",
):
    """Create the SVG elements, related to the axes."""
    rgb = ["red", "green", "blue"]

    svg_elements = []

    for i in axes[:, 2].argsort():
        a = inset[0]
        b = window_size[1] - inset[1]
        c = int(axes[i][0] * length + a)
        d = int(axes[i][1] * length + b)
        e = int(axes[i][0] * length * font_offset + a)
        f = int(axes[i][1] * length * font_offset + b)
        svg_elements.append(
            shapes.Line([a, b], [c, d], stroke=line_color, stroke_width=line_width)
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


def create_svg_document(
    elements, size, viewbox=None, background_color="white", background_opacity=1.0
):
    """Create the full SVG document.

    :param viewbox: (minx, miny, width, height)
    """
    dwg = Drawing("ase.svg", profile="tiny", size=size)
    root = Group(id="root")
    dwg.add(root)
    # if Color(background_color).web != "white":
    # apparently the best way, see: https://stackoverflow.com/a/11293812/5033292
    root.add(
        shapes.Rect(size=size, fill=background_color, fill_opacity=background_opacity)
    )
    for element in elements:
        root.add(element)
    if viewbox:
        dwg.viewbox(*viewbox)
    return dwg


def create_svg_document_with_light(
    elements, size, viewbox=None, background_color="white", background_opacity=1.0
):
    """Create the full SVG document, with a lighting filter.

    Resources:

    - https://www.w3.org/TR/SVG11/filters.html#LightSourceDefinitions
    - https://svgwrite.readthedocs.io/en/master/classes/filters.html
    - http://www.svgbasics.com/filters2.html
    - https://css-tricks.com/look-svg-light-source-filters/

    :param viewbox: (minx, miny, width, height)
    """
    # TODO work in progress
    # TODO have a look at how threejs is converted to SVG:
    # https://github.com/mrdoob/three.js/blob/master/examples/jsm/renderers/SVGRenderer.js
    dwg = Drawing("ase.svg", profile="full", size=size)

    light_filter = dwg.defs.add(Filter(size=("100%", "100%")))
    diffuse_lighting = light_filter.feDiffuseLighting(
        size=size, surfaceScale=10, diffuseConstant=1, kernelUnitLength=1, color="white"
    )
    diffuse_lighting.fePointLight(source=(size[0], 0, 1000))
    light_filter.feComposite(operator="arithmetic", k1=1)

    root = Group(id="root", filter=light_filter.get_funciri())
    dwg.add(root)
    # if Color(background_color).web != "white":
    # apparently the best way, see: https://stackoverflow.com/a/11293812/5033292
    root.add(
        shapes.Rect(size=size, fill=background_color, fill_opacity=background_opacity)
    )
    for element in elements:
        root.add(element)
    if viewbox:
        dwg.viewbox(*viewbox)
    return dwg


def string_to_compose(string):
    """Convert an SVG string to a ``svgutils.compose.SVG``."""
    from svgutils.compose import SVG
    from svgutils.transform import fromstring

    svg_figure = fromstring(string)
    element = SVG()
    element.root = svg_figure.getroot().root
    return element, list(map(float, svg_figure.get_size()))


def tessellate_rectangles(sizes, max_columns=None):
    """Compute the minimum size grid, required to fit a list of rectangles."""
    original_length = len(sizes)
    sizes = np.array(sizes, dtype=float)

    max_columns = min(max_columns, len(sizes)) if max_columns else len(sizes)

    overflow = sizes.shape[0] % max_columns
    empty = max_columns - overflow if overflow else 0
    sizes = np.concatenate((sizes, np.full([empty, 2], np.nan)))

    if len(sizes.shape) == 2:
        sizes = sizes.reshape((sizes.shape[0], 1, 2))

    sizes = np.reshape(sizes, (int(len(sizes) / max_columns), max_columns, 2))

    heights = sizes[:, :, 1]
    widths = sizes[:, :, 0]

    xv, yv = np.meshgrid(np.nanmax(widths, axis=0), np.nanmax(heights, axis=1))

    max_width = np.nanmax(np.cumsum(xv, axis=1))
    max_height = np.nanmax(np.cumsum(yv, axis=0))

    wposition = (np.cumsum(xv, axis=1) - xv).flatten()
    hposition = (np.cumsum(yv, axis=0) - yv).flatten()

    return [max_width, max_height], list(zip(wposition, hposition))[:original_length]


def get_svg_string(svg):
    """Return the raw string of an SVG object with a ``tostring`` or ``to_str`` method."""
    if isinstance(svg, str):
        return svg
    if hasattr(svg, "tostring"):
        # svgwrite.drawing.Drawing.tostring()
        return svg.tostring()
    if hasattr(svg, "to_str"):
        # svgutils.transform.SVGFigure.to_str()
        return svg.to_str()
    raise TypeError(f"SVG cannot be converted to a raw string: {svg}")


def concatenate_svgs(
    svgs,
    max_columns=None,
    scale=None,
    label=False,
    size=12,
    weight="bold",
    inset=(0.1, 0.1),
):
    """Create a grid of SVGs, with a maximum number of columns.

    Parameters
    ----------
    svgs : list
        Items may be raw SVG strings,
        or any objects with a ``tostring`` or ``to_str`` method.
    max_columns : int or None
        max number of columns, or if None, only use one row
    scale : float or None
        scale the entire composition
    label : bool
        whether to add a label for each SVG (cycle through upper case letters)
    size : int
        label font size
    weight : str
        label font weight
    inset : tuple
        inset the label by x times the SVG width and y times the SVG height

    Returns
    -------
    svgutils.compose.Figure

    """
    # TODO could replace svgutils with use of lxml primitives
    from svgutils.compose import Figure, Text

    label_iter = cycle("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    svg_composes, dimensions = zip(
        *[string_to_compose(get_svg_string(svg)) for svg in svgs]
    )
    if scale:
        [svg.scale(scale) for svg in svg_composes]
        dimensions = [(w * scale, h * scale) for (w, h) in dimensions]
    (width, height), positions = tessellate_rectangles(dimensions, max_columns)
    elements = []
    for svg, (x, y), (w, h) in zip(svg_composes, positions, dimensions):
        elements.append(svg.move(x, y))
        if label:
            elements.append(
                Text(
                    next(label_iter),
                    x=x + inset[0] * w,
                    y=y + inset[1] * h,
                    size=size,
                    weight=weight,
                )
            )
    return Figure(width, height, *elements)


def svg_to_pdf(svg, file_name=None):
    """Convert SVG to PDF.

    To view in notebook::

        from IPython.display import display_pdf

        rlg_drawing = svg_to_pdf(svg)
        display_pdf(rlg_drawing.asString("pdf"), raw=True)

    """
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

    string = get_svg_string(svg)
    fd, fname = tempfile.mkstemp()
    try:
        with open(fname, "w") as handle:
            handle.write(string)
        rlg_drawing = svg2rlg(fname)
    finally:
        if os.path.exists(fname):
            os.remove(fname)

    if file_name:
        renderPDF.drawToFile(rlg_drawing, file_name)

    return rlg_drawing
