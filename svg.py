"""A module for creating an SVG visualisation of a structure."""
from svgwrite import Drawing, shapes, text  # noqa F401


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
