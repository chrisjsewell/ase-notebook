"""Tests for ``aiida_2d.visualize.svg``."""
from aiida_2d.visualize.svg import tessellate_rectangles
from aiida_2d.visualize.viewer import AseView


def test_tessellate_rectangles():
    """Test tessellate_rectangles."""
    sizes = [[5, 2], [3, 4], [5, 9], [7, 8], [9, 10]]

    total_size, positions = tessellate_rectangles(sizes, 1)
    assert total_size == [9.0, 33.0]
    assert positions == [(0.0, 0.0), (0.0, 2.0), (0.0, 6.0), (0.0, 15.0), (0.0, 23.0)]

    total_size, positions = tessellate_rectangles(sizes, 2)
    assert total_size == [16.0, 23.0]
    assert positions == [(0.0, 0.0), (9.0, 0.0), (0.0, 4.0), (9.0, 4.0), (0.0, 13.0)]

    total_size, positions = tessellate_rectangles(sizes, 3)
    assert total_size == [21.0, 19.0]
    assert positions == [(0.0, 0.0), (7.0, 0.0), (16.0, 0.0), (0.0, 9.0), (7.0, 9.0)]

    total_size, positions = tessellate_rectangles(sizes, 4)
    assert total_size == [24.0, 19.0]
    assert positions == [(0.0, 0.0), (9.0, 0.0), (12.0, 0.0), (17.0, 0.0), (0.0, 9.0)]

    total_size, positions = tessellate_rectangles(sizes, 5)
    assert total_size == [29.0, 10.0]
    assert positions == [(0.0, 0.0), (5.0, 0.0), (8.0, 0.0), (13.0, 0.0), (20.0, 0.0)]

    total_size, positions = tessellate_rectangles(sizes, 6)
    assert total_size == [29.0, 10.0]
    assert positions == [(0.0, 0.0), (5.0, 0.0), (8.0, 0.0), (13.0, 0.0), (20.0, 0.0)]

    total_size, positions = tessellate_rectangles(sizes, None)
    assert total_size == [29.0, 10.0]
    assert positions == [(0.0, 0.0), (5.0, 0.0), (8.0, 0.0), (13.0, 0.0), (20.0, 0.0)]


def test_make_svg(get_test_structure):
    """Test make svg, from the viewer."""
    structure = get_test_structure("pyrite")
    structure.add_site_property("ghost", [True] + [False for _ in structure][1:])
    ase_view = AseView(
        uc_segments=2,
        show_bonds=True,
        element_colors="vesta",
        element_radii="vesta",
        rotations="45x,45y,45z",
        atom_font_size=16,
        show_axes=True,
        axes_length=30,
        canvas_size=(400, 400),
        zoom=1.2,
    )
    ase_view.add_miller_plane(1, 2, 1, color="lightgreen", stroke_width=1, as_poly=True)
    svg = ase_view.make_svg(structure, center_in_uc=True)
    assert len(svg.elements) == 2


def test_make_svg_occupancies(get_test_structure):
    """Test ``make_svg``, from the viewer, when element occupacies are specified."""
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = get_test_structure("pyrite")
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.info["occupancy"] = {0: {"Fe": 0.75, "S": 0.1, "H": 0.1}}
    ase_view = AseView(
        uc_segments=2,
        show_bonds=True,
        element_colors="vesta",
        element_radii="vesta",
        rotations="45x,45y,45z",
        atom_font_size=16,
        show_axes=True,
        axes_length=30,
        canvas_size=(400, 400),
        zoom=1.2,
    )
    ase_view.add_miller_plane(1, 2, 1, color="lightgreen", stroke_width=1, as_poly=True)
    svg = ase_view.make_svg(structure, center_in_uc=True)
    assert len(svg.elements) == 2
