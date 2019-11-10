Changelog
=========

Version 0.3.1
-------------

Improve documentation and add Binder build.

Version 0.3.0
-------------
- Add autodoc decorator.

- Restructure and Add CI Testing (#1)

  - Restructured code and added more tests
  - Added development setting and pre-commit
  - Added package setup
  - Added atoms_convert and remove pymatgen as dependency.
- Move test modules to folder.

- Merge aiida-2d/tests/test_visualize from branch 'develop'

  This allows the files to be copied to the new repository,
  whilst retaining the commit history.
- Add dash pattern config and pythreejs render.

- Directly use ViewConfig in AtomGui.

- Visualisation implement more options in GUI.

- Add svg concatenation.

- Tidy visualize module.

- Move GUI launching to AseView.

- Abstract out part of gui functionality.

- Add visualisation of miller planes as polygon.

- Add thickness to miller index planes.

- Add conversion of canvas to SVG.

- Merge aiida-2d/aiida_2d/visualize from branch 'develop'

  This allows the code to be moved to this new repository,
  whilst retaining the commit history.
- Move files into package.

- Atom atom picking, to show information.

- Added labels to threejs.

- Threejs: allow use of ``CloneArray`` and add basic gui.

- Add create_svg_document_with_light.

- Three: add render container and improve camera distance.

- Improve number validation and add background opacity.

- Add miller_lines to threejs render.

- Color/label by arrays.

- Make the miller plane configuration an ``attr`` class.

- Split up AseView._initialise_elements, and implement pythreejs world axes.

- Add dash pattern config and pythreejs render.

- Directly use ViewConfig in AtomGui.

- Add fractional occupancy to svg draw.

- Visualisation implement more options in GUI.

- Extract the drawing functionality from the draw method.

- Refactored AtomGui.draw to utilise DrawGroup.

- Visualise: implement additional color configuration.

- Add ``svg_to_pdf`` function.

- Add svg concatenation.

- Tidy visualize module.

- Move GUI launching to AseView.

- Improve SVG creation.

- Add svg creation.

- Add implementation agnostic visualisation element classes.

- Abstract out part of gui functionality.

- Visualisation add unit_cell_segmentation and atom opacity.

- Add visualisation of miller planes as polygon.

- Extract _draw_atom into separate function.

- Refactor AtomGui.set_atoms and AtomGui.draw, to be easier to understand.

- Allow for opacity to be set in SVG (via the use of tkinter element tags)

- Add thickness to miller index planes.

- Add conversion of canvas to SVG.

- Initial commit.
