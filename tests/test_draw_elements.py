"""Module for testing ``ase_notebook.draw_elements``."""
import ase_notebook.draw_elements as draw


def test_element():
    """Test single Element initialisation."""
    el = draw.Element(a=1, b=2)
    assert {k: el[k] for k in el} == {"a": 1, "b": 2}
    assert el.a == 1
    assert el.b == 2


def test_draw_elements():
    """Test initialisation and representation."""
    spheres = draw.DrawElementsSphere(
        "atoms",
        coordinates=[[0, 0, 0], [1, 1, 1]],
        radii=[1, 1],
        element_properties={"color": ["red", "blue"]},
        group_properties={"other": "a"},
    )
    assert str(spheres) == (
        "DrawElementsSphere(name=atoms, elements=2, "
        "el_properties=(color), grp_properties=(other))"
    )

    assert spheres[0].color == "red"

    lines = draw.DrawElementsLine(
        "lines",
        [[[0, 0, 0], [1, 1, 1]]],
        element_properties={"thickness": [1]},
        group_properties={"other2": "a"},
    )
    assert str(lines) == (
        "DrawElementsLine(name=lines, elements=1, "
        "el_properties=(thickness), grp_properties=(other2))"
    )

    assert lines[0]["other2"] == "a"

    polys = draw.DrawElementsPoly("polys", [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    assert str(polys) == (
        "DrawElementsPoly(name=polys, elements=1, "
        "el_properties=(), grp_properties=())"
    )

    group = draw.DrawGroup([spheres, lines, polys])
    assert str(group) == "DrawGroup(groups=(atoms, lines, polys), elements=(2, 1, 1))"
