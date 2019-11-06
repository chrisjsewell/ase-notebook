"""A module for creating a pythreejs visualisation of a structure."""
import numpy as np

from aiida_2d.visualize import Color


def triangle_normal(a, b, c):
    """Compute the normal of three points."""
    a, b, c = [np.array(i) for i in (a, b, c)]
    return np.cross(b - a, c - a).tolist()


def generate_3js_render(element_groups, canvas_size, zoom, camera_fov=30):
    """Create a pythreejs scene of the elements."""
    import pythreejs as pjs

    group_elem = pjs.Group()

    sphere_geom = {}
    sphere_mat = {}

    outline_material = pjs.MeshBasicMaterial(
        color="black", side="BackSide", transparent=True
    )
    # label_texture = pjs.TextTexture(string="Fe", color="black")
    # label_material = pjs.MeshLambertMaterial(
    #     transparent=True, opacity=1.0, map=label_texture
    # )

    for el in element_groups["atoms"]:
        if el.sradius not in sphere_geom:
            sphere_geom[el.sradius] = pjs.SphereBufferGeometry(
                radius=el.sradius, widthSegments=30, heightSegments=30
            )
        material_hash = (el.color, el.fill_opacity)
        if material_hash not in sphere_mat:
            sphere_mat[material_hash] = pjs.MeshLambertMaterial(
                color=el.color, transparent=True, opacity=el.fill_opacity
            )
        mesh = pjs.Mesh(
            geometry=sphere_geom[el.sradius], material=sphere_mat[material_hash]
        )
        mesh.position = el.position.tolist()
        group_elem.add(mesh)

        outline_mesh = pjs.Mesh(
            geometry=sphere_geom[el.sradius], material=outline_material
        )
        outline_mesh.position = el.position.tolist()
        outline_mesh.scale = (np.array(outline_mesh.scale) * 1.05).tolist()
        group_elem.add(outline_mesh)

    #     label_mesh = pjs.Mesh(geometry=sphere_geom[el.sradius], material=label_material)
    #     label_mesh.position = el.position.tolist()
    #     group_elem.add(label_mesh)

    cell_line_mat = pjs.LineMaterial(
        linewidth=1, color=element_groups["cell_lines"].group_properties["color"]
    )
    cell_line_geo = pjs.LineSegmentsGeometry(
        positions=[el.position.tolist() for el in element_groups["cell_lines"]]
    )
    cell_lines = pjs.LineSegments2(geometry=cell_line_geo, material=cell_line_mat)
    group_elem.add(cell_lines)

    bond_line_mat = pjs.LineMaterial(
        linewidth=element_groups["bond_lines"].group_properties["stroke_width"],
        vertexColors="VertexColors",
    )
    bond_line_geo = pjs.LineSegmentsGeometry(
        positions=[el.position.tolist() for el in element_groups["bond_lines"]],
        colors=[
            [Color(c).rgb for c in el.color] for el in element_groups["bond_lines"]
        ],
    )
    bond_lines = pjs.LineSegments2(geometry=bond_line_geo, material=bond_line_mat)
    group_elem.add(bond_lines)

    for el in element_groups["miller_planes"]:
        vertices = el.position.tolist()
        faces = [
            (
                0,
                1,
                2,
                triangle_normal(vertices[0], vertices[1], vertices[2]),
                "black",
                0,
            )
        ]
        if len(vertices) == 4:
            faces.append(
                (
                    2,
                    3,
                    0,
                    triangle_normal(vertices[2], vertices[3], vertices[0]),
                    "black",
                    0,
                )
            )
        elif len(vertices) != 3:
            raise NotImplementedError("polygons with more than 4 points")
        plane_geom = pjs.Geometry(vertices=vertices, faces=faces)
        plane_mat = pjs.MeshBasicMaterial(
            color=el.color, transparent=True, opacity=el.fill_opacity, side="DoubleSide"
        )
        plane_mesh = pjs.Mesh(geometry=plane_geom, material=plane_mat)
        group_elem.add(plane_mesh)

    scene = pjs.Scene()
    scene.add([group_elem])

    minp, maxp = element_groups.get_position_range()

    view_width, view_height = canvas_size

    minp, maxp = element_groups.get_position_range()
    pos_camera = minp[2] - (maxp[2] - minp[2])

    camera = pjs.PerspectiveCamera(
        fov=camera_fov,
        position=[0, 0, pos_camera],
        aspect=view_width / view_height,
        zoom=zoom,
    )
    scene.add([camera])
    ambient_light = pjs.AmbientLight(color="lightgray")
    direct_light = pjs.DirectionalLight(position=(maxp * 2).tolist())
    scene.add([camera, ambient_light, direct_light])

    controller = pjs.OrbitControls(controlling=camera, screenSpacePanning=True)
    renderer = pjs.Renderer(
        camera=camera,
        scene=scene,
        controls=[controller],
        width=view_width,
        height=view_height,
    )
    return renderer


def create_world_axes(camera, controls, initial_rotation=np.eye(3), length=50, width=3):
    """Create a renderer, containing an axes and camera that is synced to another camera.

    adapted from http://jsfiddle.net/aqnL1mx9/

    Parameters
    ----------
    camera : pythreejs.PerspectiveCamera
    controls : pythreejs.OrbitControls
    initial_rotation : list or numpy.array
        initial rotation of the axes
    length : int
        length of axes lines
    width : int
        line width of axes

    Returns
    -------
    pythreejs.Renderer

    """
    import pythreejs as pjs

    canvas_width = length * 2
    canvas_height = length * 2
    cam_distance = length * 3

    ax_scene = pjs.Scene()

    group_ax = pjs.Group()
    ax_line_mat = pjs.LineMaterial(linewidth=width, vertexColors="VertexColors")
    ax_line_geo = pjs.LineSegmentsGeometry(
        positions=[
            [[0, 0, 0], length * r / np.linalg.norm(r)] for r in initial_rotation
        ],
        colors=[[Color(c).rgb] * 2 for c in ("red", "green", "blue")],
    )
    ax_lines = pjs.LineSegments2(geometry=ax_line_geo, material=ax_line_mat)
    group_ax.add(ax_lines)

    ax_scene.add([group_ax])

    ax_camera = pjs.PerspectiveCamera(
        fov=50, aspect=canvas_width / canvas_height, near=1, far=1000
    )
    ax_camera.up = camera.up
    ax_renderer = pjs.Renderer(
        scene=ax_scene, camera=ax_camera, width=canvas_width, height=canvas_height
    )

    def align_axes(change=None):
        """Align axes to world."""
        # TODO: this is not working correctly for TrackballControls, when rotated upside-down
        # (OrbitControls enforces the camera up direction,
        # so does not allow the camera to rotate upside-down).
        # TODO how could this be implemented on the client (js) side?
        new_position = np.array(camera.position) - np.array(controls.target)
        new_position = cam_distance * new_position / np.linalg.norm(new_position)
        ax_camera.position = new_position.tolist()
        ax_camera.lookAt(ax_scene.position)

    align_axes()

    camera.observe(align_axes, names="position")
    controls.observe(align_axes, names="target")
    ax_scene.observe(align_axes, names="position")

    return ax_renderer
