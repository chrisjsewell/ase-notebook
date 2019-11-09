"""A module for creating a pythreejs visualisation of a structure."""
from math import radians, sqrt, tan

import numpy as np

from ase_notebook.color import Color
from ase_notebook.draw_utils import triangle_normal


class RenderContainer(object):
    """Container for the renderer, with attribute access for key elements."""

    def __init__(self, top_level, **kwargs):
        """Initialise container."""
        self._kwargs = kwargs
        self.top_level = top_level

    def __dir__(self):
        """Get the attributes."""
        return list(self._kwargs.keys())

    def __getitem__(self, key):
        """Return key."""
        return self._kwargs[key]

    def __setitem__(self, key, value):
        """Set key."""
        if key == "top_level":
            self.top_level = value
        else:
            self._kwargs[key] = value

    def __getattr__(self, key):
        """Return attribute."""
        if key not in self._kwargs:
            raise AttributeError(key)
        return self._kwargs[key]

    def __setattr__(self, name, value):
        """Set attribute."""
        if name == "top_level":
            if not hasattr(value, "_ipython_display_"):
                raise ValueError("top_level must have an `_ipython_display_` method")
            self._kwargs["top_level"] = value
            return
        if name != "_kwargs":
            raise AttributeError("Attributes are frozen")
        return super().__setattr__(name, value)

    def __contains__(self, key):
        """Test if key in container."""
        return key in self._kwargs

    def _ipython_display_(self):
        """Display the top level rendered in the notebook."""
        return self.top_level._ipython_display_()


def generate_3js_render(
    element_groups,
    canvas_size,
    zoom,
    camera_fov=30,
    background_color="white",
    background_opacity=1.0,
    reuse_objects=False,
    use_atom_arrays=False,
    use_label_arrays=False,
):
    """Create a pythreejs scene of the elements.

    Regarding initialisation performance, see: https://github.com/jupyter-widgets/pythreejs/issues/154
    """
    import pythreejs as pjs

    key_elements = {}
    group_elements = pjs.Group()
    key_elements["group_elements"] = group_elements

    unique_atom_sets = {}
    for el in element_groups["atoms"]:
        element_hash = (
            ("radius", el.sradius),
            ("color", el.color),
            ("fill_opacity", el.fill_opacity),
            ("stroke_color", el.get("stroke_color", "black")),
            ("ghost", el.ghost),
        )
        unique_atom_sets.setdefault(element_hash, []).append(el)

    group_atoms = pjs.Group()
    group_ghosts = pjs.Group()

    atom_geometries = {}
    atom_materials = {}
    outline_materials = {}

    for el_hash, els in unique_atom_sets.items():
        el = els[0]
        data = dict(el_hash)

        if reuse_objects:
            atom_geometry = atom_geometries.setdefault(
                el.sradius,
                pjs.SphereBufferGeometry(
                    radius=el.sradius, widthSegments=30, heightSegments=30
                ),
            )
        else:
            atom_geometry = pjs.SphereBufferGeometry(
                radius=el.sradius, widthSegments=30, heightSegments=30
            )

        if reuse_objects:
            atom_material = atom_materials.setdefault(
                (el.color, el.fill_opacity),
                pjs.MeshLambertMaterial(
                    color=el.color, transparent=True, opacity=el.fill_opacity
                ),
            )
        else:
            atom_material = pjs.MeshLambertMaterial(
                color=el.color, transparent=True, opacity=el.fill_opacity
            )

        if use_atom_arrays:
            atom_mesh = pjs.Mesh(geometry=atom_geometry, material=atom_material)
            atom_array = pjs.CloneArray(
                original=atom_mesh,
                positions=[e.position.tolist() for e in els],
                merge=False,
            )
        else:
            atom_array = [
                pjs.Mesh(
                    geometry=atom_geometry,
                    material=atom_material,
                    position=e.position.tolist(),
                    name=e.info_string,
                )
                for e in els
            ]

        data["geometry"] = atom_geometry
        data["material_body"] = atom_material

        if el.ghost:
            key_elements["group_ghosts"] = group_ghosts
            group_ghosts.add(atom_array)
        else:
            key_elements["group_atoms"] = group_atoms
            group_atoms.add(atom_array)

        if el.get("stroke_width", 1) > 0:
            if reuse_objects:
                outline_material = outline_materials.setdefault(
                    el.get("stroke_color", "black"),
                    pjs.MeshBasicMaterial(
                        color=el.get("stroke_color", "black"),
                        side="BackSide",
                        transparent=True,
                        opacity=el.get("stroke_opacity", 1.0),
                    ),
                )
            else:
                outline_material = pjs.MeshBasicMaterial(
                    color=el.get("stroke_color", "black"),
                    side="BackSide",
                    transparent=True,
                    opacity=el.get("stroke_opacity", 1.0),
                )
            # TODO use stroke width to dictate scale
            if use_atom_arrays:
                outline_mesh = pjs.Mesh(
                    geometry=atom_geometry,
                    material=outline_material,
                    scale=(1.05, 1.05, 1.05),
                )
                outline_array = pjs.CloneArray(
                    original=outline_mesh,
                    positions=[e.position.tolist() for e in els],
                    merge=False,
                )
            else:
                outline_array = [
                    pjs.Mesh(
                        geometry=atom_geometry,
                        material=outline_material,
                        position=e.position.tolist(),
                        scale=(1.05, 1.05, 1.05),
                    )
                    for e in els
                ]

            data["material_outline"] = outline_material

            if el.ghost:
                group_ghosts.add(outline_array)
            else:
                group_atoms.add(outline_array)

        key_elements.setdefault("atom_arrays", []).append(data)

    group_elements.add(group_atoms)
    group_elements.add(group_ghosts)

    group_labels = add_labels(element_groups, key_elements, use_label_arrays)
    group_elements.add(group_labels)

    if len(element_groups["cell_lines"]) > 0:
        cell_line_mat = pjs.LineMaterial(
            linewidth=1, color=element_groups["cell_lines"].group_properties["color"]
        )
        cell_line_geo = pjs.LineSegmentsGeometry(
            positions=[el.position.tolist() for el in element_groups["cell_lines"]]
        )
        cell_lines = pjs.LineSegments2(geometry=cell_line_geo, material=cell_line_mat)
        key_elements["cell_lines"] = cell_lines
        group_elements.add(cell_lines)

    if len(element_groups["bond_lines"]) > 0:
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
        key_elements["bond_lines"] = bond_lines
        group_elements.add(bond_lines)

    group_millers = pjs.Group()
    if len(element_groups["miller_lines"]) or len(element_groups["miller_planes"]):
        key_elements["group_millers"] = group_millers

    if len(element_groups["miller_lines"]) > 0:
        miller_line_mat = pjs.LineMaterial(
            linewidth=3, vertexColors="VertexColors"  # TODO use stroke_width
        )
        miller_line_geo = pjs.LineSegmentsGeometry(
            positions=[el.position.tolist() for el in element_groups["miller_lines"]],
            colors=[
                [Color(el.stroke_color).rgb] * 2
                for el in element_groups["miller_lines"]
            ],
        )
        miller_lines = pjs.LineSegments2(
            geometry=miller_line_geo, material=miller_line_mat
        )
        group_millers.add(miller_lines)

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
            color=el.fill_color,
            transparent=True,
            opacity=el.fill_opacity,
            side="DoubleSide",
        )
        plane_mesh = pjs.Mesh(geometry=plane_geom, material=plane_mat)
        group_millers.add(plane_mesh)

    group_elements.add(group_millers)

    scene = pjs.Scene(background=None)
    scene.add([group_elements])

    view_width, view_height = canvas_size

    minp, maxp = element_groups.get_position_range()
    # compute a minimum camera distance, that is guaranteed to encapsulate all elements
    camera_dist = maxp[2] + sqrt(maxp[0] ** 2 + maxp[1] ** 2) / tan(
        radians(camera_fov / 2)
    )

    camera = pjs.PerspectiveCamera(
        fov=camera_fov,
        position=[0, 0, camera_dist],
        aspect=view_width / view_height,
        zoom=zoom,
    )
    scene.add([camera])
    ambient_light = pjs.AmbientLight(color="lightgray")
    key_elements["ambient_light"] = ambient_light
    direct_light = pjs.DirectionalLight(position=(maxp * 2).tolist())
    key_elements["direct_light"] = direct_light
    scene.add([camera, ambient_light, direct_light])

    camera_control = pjs.OrbitControls(controlling=camera, screenSpacePanning=True)

    atom_picker = pjs.Picker(controlling=group_atoms, event="dblclick")
    key_elements["atom_picker"] = atom_picker
    material = pjs.SpriteMaterial(
        map=create_arrow_texture(right=False),
        transparent=True,
        depthWrite=False,
        depthTest=False,
    )
    atom_pointer = pjs.Sprite(material=material, scale=(4, 3, 1), visible=False)
    scene.add(atom_pointer)
    key_elements["atom_pointer"] = atom_pointer

    renderer = pjs.Renderer(
        camera=camera,
        scene=scene,
        controls=[camera_control, atom_picker],
        width=view_width,
        height=view_height,
        alpha=True,
        clearOpacity=background_opacity,
        clearColor=background_color,
    )
    return renderer, key_elements


def add_labels(element_groups, key_elements, use_label_arrays):
    """Create label elements for the scene."""
    import pythreejs as pjs

    group_labels = pjs.Group()
    unique_label_sets = {}

    for el in element_groups["atoms"]:
        if "label" in el and el.label is not None:
            unique_label_sets.setdefault(
                (("label", el.label), ("color", el.get("font_color", "black"))), []
            ).append(el)

    if unique_label_sets:
        key_elements["group_labels"] = group_labels

    for el_hash, els in unique_label_sets.items():
        el = els[0]
        data = dict(el_hash)
        # depthWrite=depthTest=False is required, for the sprite to remain on top,
        # and not have the whitespace obscure objects behind, see:
        # https://stackoverflow.com/questions/11165345/three-js-webgl-transparent-planes-hiding-other-planes-behind-them
        # TODO can this be improved?
        text_material = pjs.SpriteMaterial(
            map=pjs.TextTexture(
                string=el.label,
                color=el.get("font_color", "black"),
                size=2000,  # this texttexture size seems to work, not sure why?
            ),
            opacity=1.0,
            transparent=True,
            depthWrite=False,
            depthTest=False,
        )
        data["material"] = text_material
        key_elements.setdefault("label_arrays", []).append(data)
        if use_label_arrays:
            text_sprite = pjs.Sprite(material=text_material)
            label_array = pjs.CloneArray(
                original=text_sprite,
                positions=[e.position.tolist() for e in els],
                merge=False,
            )
        else:
            label_array = [
                pjs.Sprite(material=text_material, position=e.position.tolist())
                for e in els
            ]
        group_labels.add(label_array)

    return group_labels


def create_world_axes(
    camera, controls, initial_rotation=np.eye(3), length=30, width=3, camera_fov=10
):
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

    ax_scene = pjs.Scene()

    group_ax = pjs.Group()
    # NOTE: could use AxesHelper, but this does not allow for linewidth seletion
    # TODO: add arrow heads (ArrowHelper doesn't seem to work)
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

    camera_dist = length / tan(radians(camera_fov / 2))

    ax_camera = pjs.PerspectiveCamera(
        fov=camera_fov, aspect=canvas_width / canvas_height, near=1, far=1000
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
        new_position = camera_dist * new_position / np.linalg.norm(new_position)
        ax_camera.position = new_position.tolist()
        ax_camera.lookAt(ax_scene.position)

    align_axes()

    camera.observe(align_axes, names="position")
    controls.observe(align_axes, names="target")
    ax_scene.observe(align_axes, names="position")

    return ax_renderer


def make_basic_gui(container):
    """Create a basic GUI layout.

    Parameters
    ----------
    container : RenderContainer

    Returns
    -------
    ipywidgets.GridspecLayout

    """
    import ipywidgets as ipyw

    element_controls = [
        ipyw.HTML(value="<b>Elements</b>", layout=ipyw.Layout(align_self="center"))
    ]
    for key, descript in [
        ("group_atoms", "Atoms"),
        ("cell_lines", "Unit Cell"),
        ("group_labels", "Labels"),
        ("bond_lines", "Bonds"),
        ("group_millers", "Planes"),
        ("group_ghosts", "Ghosts"),
    ]:
        if key not in container:
            continue
        toggle = ipyw.ToggleButton(
            description=descript,
            icon="eye",
            button_style="primary",
            value=False if key == "group_labels" else container[key].visible,
            layout=ipyw.Layout(width="auto"),
        )
        ipyw.jslink((toggle, "value"), (container[key], "visible"))
        element_controls.append(toggle)

    control_box_elements = ipyw.Box(
        element_controls, layout=ipyw.Layout(flex_flow="column")
    )
    container["control_box_elements"] = control_box_elements

    background_controls = [
        ipyw.HTML(value="<b>Background</b>", layout=ipyw.Layout(align_self="center"))
    ]
    background_color = ipyw.ColorPicker(
        concise=True,
        description="Color",
        description_tooltip="Background Color",
        value=container.element_renderer.clearColor,
        layout=ipyw.Layout(align_items="center"),
    )
    background_color.style.description_width = "40px"
    ipyw.jslink((background_color, "value"), (container.element_renderer, "clearColor"))
    background_controls.append(background_color)
    background_opacity = ipyw.FloatSlider(
        value=container.element_renderer.clearOpacity,
        min=0,
        max=1,
        step=0.1,
        orientation="horizontal",
        readout=False,
        description_tooltip="Background Opacity",
    )
    background_opacity.layout.max_width = "100px"
    ipyw.jslink(
        (background_opacity, "value"), (container.element_renderer, "clearOpacity")
    )
    background_controls.append(background_opacity)
    # other_controls.append(ipyw.Label(value="Opacity", layout=ipyw.Layout(align_self="center")))

    control_box_background = ipyw.Box(
        background_controls, layout=ipyw.Layout(flex_flow="column")
    )
    container["control_box_background"] = control_box_background

    axes = [container.axes_renderer] if "axes_renderer" in container else []

    info_box = ipyw.HTML(
        value="Double-click atom for info.",
        color="grey",
        layout=ipyw.Layout(
            max_height="10px", margin="0px 0px 0px 0px", align_self="flex-start"
        ),
    )

    def on_click(change):
        obj = change["new"]
        if obj is None:
            container.atom_pointer.visible = False
            info_box.value = ""
        else:
            info_box.value = obj.name
            # container.atom_pointer.position = container.atom_picker.point
            container.atom_pointer.position = obj.position
            container.atom_pointer.visible = True

    container.atom_picker.observe(on_click, names=["object"])

    if axes and container.element_renderer.height > 200:
        grid = ipyw.GridspecLayout(
            2,
            2,
            width=f"{container.element_renderer.width + 100}px",
            height=f"{container.element_renderer.height + 35}px",
        )
        grid[0, 0] = container.element_renderer
        grid[1, 0] = info_box
        grid[:, 1] = ipyw.Box(
            axes + [control_box_elements, control_box_background],
            layout=ipyw.Layout(align_self="flex-start", flex_flow="column"),
        )
    else:
        grid = ipyw.GridspecLayout(
            2,
            3,
            width=f"{container.element_renderer.width + 200}px",
            height=f"{container.element_renderer.height + 35}px",
        )
        grid[:, 0] = ipyw.Box(
            axes, layout=ipyw.Layout(align_self="flex-end", flex_flow="column")
        )
        grid[0, 1] = container.element_renderer
        grid[1, 1] = info_box
        grid[:, 2] = ipyw.Box(
            [control_box_elements, control_box_background],
            layout=ipyw.Layout(align_self="flex-start", flex_flow="column"),
        )

    return grid


def gather_3d_objects(obj, objects=None):
    """Recurse through objects children, to gather the set of 3D objects."""
    # TODO create more complete method
    import pythreejs as pjs

    if objects is None:
        objects = set()
    if isinstance(obj, pjs.Renderer):
        gather_3d_objects(obj.scene, objects)
    elif isinstance(obj, pjs.Scene):
        for child in obj.children:
            gather_3d_objects(child, objects)
    elif isinstance(obj, pjs.Object3DBase):
        objects.add(obj)
        for child in obj.children:
            gather_3d_objects(child, objects)
        if "geometry" in obj.trait_names():
            objects.add(obj.geometry)
        if "material" in obj.trait_names():
            objects.add(obj.material)
        if isinstance(obj, pjs.CloneArray):
            gather_3d_objects(obj.original, objects)

    return objects


def create_arrow_texture(width=2 ** 9, height=2 ** 9, color="red", right=True):
    """Create an array map of an arrow."""
    import pythreejs as pjs

    color_rgba = list(Color(color).rgb) + [1.0]

    array = np.zeros((width, height, 4), dtype="float32")
    if right:  # facing right
        for y in range(0, int(width / 4)):
            for x in range(int(height * 9 / 24), int(height * 15 / 24)):
                array[x, y, :] = color_rgba
        for y in range(int(width / 4), int(width / 2)):
            for x in range(
                int(0 + height * y / (width)), int(height - height * y / (width))
            ):
                array[x, y, :] = color_rgba
    else:  # facing left
        for y in range(int(width * 3 / 4), int(width)):
            for x in range(int(height * 9 / 24), int(height * 15 / 24)):
                array[x, y, :] = color_rgba
        for y in range(int(width / 2), int(width * 3 / 4)):
            for x in range(
                int(height - ((height / 2) * y / (width / 2))),
                int(0 + ((height / 2) * y / (width / 2))),
            ):
                array[x, y, :] = color_rgba

    return pjs.DataTexture(data=array, format="RGBAFormat", type="FloatType")
