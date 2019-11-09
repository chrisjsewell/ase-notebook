"""Tests for ``ase_notebook.threejs``."""
from ase_notebook.backend import threejs
from ase_notebook.viewer import AseView


def test_make_render(get_test_atoms):
    """Test make svg, from the viewer."""
    atoms = get_test_atoms("pyrite")
    ase_view = AseView(
        uc_dash_pattern=(0.6, 0.4),
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
    ase_view.add_miller_plane(1, 2, 1, color="lightgreen", stroke_width=1)
    container = ase_view.make_render(atoms, center_in_uc=True)
    assert list(container) == [
        "element_renderer",
        "group_elements",
        "group_atoms",
        "atom_arrays",
        "group_labels",
        "label_arrays",
        "cell_lines",
        "bond_lines",
        "group_millers",
        "ambient_light",
        "direct_light",
        "atom_picker",
        "atom_pointer",
        "axes_renderer",
        "top_level",
        "control_box_elements",
        "control_box_background",
    ]


def test_create_arrow_texture():
    """Test creation of arrow texture."""
    texture = threejs.create_arrow_texture(2 ** 2, 2 ** 2, right=True)
    assert texture.data.tolist() == [
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    ]
    texture = threejs.create_arrow_texture(2 ** 2, 2 ** 2, right=False)
    assert texture.data.tolist() == [
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    ]
