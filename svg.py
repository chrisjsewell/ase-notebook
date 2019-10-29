"""A module for creating an SVG visualisation of a structure."""
from itertools import cycle
import os
import tempfile

import numpy as np
from svgwrite import Drawing, shapes, text
from svgwrite.container import Group


def generate_svg_elements(element_group, scale):
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
    axes, window_size, length=15, width=1, font_size=14, inset=(20, 20), font_offset=1.0
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


def create_svg_document(elements, size, viewbox=None):
    """Create the full SVG document.

    :param viewbox: (minx, miny, width, height)
    """
    dwg = Drawing("ase.svg", profile="tiny", size=size)
    root = Group(id="root")
    dwg.add(root)
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


def svg_to_pdf(svg, fname=None):
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

    if fname:
        renderPDF.drawToFile(rlg_drawing, fname)

    return rlg_drawing
