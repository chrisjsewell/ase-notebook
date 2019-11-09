"""Module defining backend agnostic containers for visualisation elements."""
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import List

import numpy as np


class Element(object):
    """Representation of a single element.

    Implemented as a frozen dictionary with attribute access.
    """

    def __init__(self, **kwargs):
        """Initialise element."""
        self._kwargs = kwargs

    def __dir__(self):
        """Get the attributes."""
        return list(self._kwargs.keys()) + ["get"]

    def get(self, key, default):
        """Return key or default."""
        if key in self:
            return self[key]
        else:
            return default

    def __repr__(self):
        """Represent object."""
        sig = ", ".join([f"{k}={self._kwargs[k]}" for k in sorted(self._kwargs)])
        return f"Element({sig})"

    def __getitem__(self, key):
        """Return key."""
        return self._kwargs[key]

    def __iter__(self):
        """Iterate property keys."""
        for key in self._kwargs:
            yield key

    def __getattr__(self, key):
        """Return key."""
        if key not in self._kwargs:
            raise AttributeError(str(key))
        return self._kwargs[key]

    def __setattr__(self, name, key):
        """Return key."""
        if name != "_kwargs":
            raise AttributeError("Element attributes are frozen")
        return super().__setattr__(name, key)

    def __contains__(self, key):
        """Test if key in object."""
        return key in self._kwargs


class DrawElementsBase:
    """Abstract base class to store a set of 3D-visualisation elements."""

    etype = None
    _protected_keys = ("name", "type", "position", "get")

    def __init__(
        self, name, coordinates, element_properties=None, group_properties=None
    ):
        """Initialise the element group."""
        self.name = name
        self._coordinates = coordinates
        self._positions = coordinates
        self._axes = np.identity(3)
        self._offset = np.zeros(3)
        self._el_props = {}
        self._grp_props = {}

        for key, val in (element_properties or {}).items():
            self.set_property(key, val, element=True)
        for key, val in (group_properties or {}).items():
            self.set_property(key, val, element=False)

    @property
    def element_properties(self):
        """Return per element properties."""
        output = deepcopy(self._el_props)
        output["positions"] = np.array(self._positions)
        return output

    @property
    def group_properties(self):
        """Return element group properties."""
        return deepcopy(self._grp_props)

    def set_property(self, name, value, element=False):
        """Set a group or per element property."""
        if name in self._protected_keys:
            raise KeyError(f"{name} is a protected key name")
        if element:
            if len(value) != len(self._coordinates):
                raise AssertionError(
                    f"property '{name}' does not have the same length "
                    "as the number of elements"
                )
            assert (
                name not in self._grp_props
            ), f"{name} is already set as a group property"
            self._el_props[name] = value
        else:
            assert (
                name not in self._el_props
            ), f"{name} is already set as an element property"
            self._grp_props[name] = value

    def set_property_many(self, properties, element=False):
        """Set multiple group or per element properties."""
        for key, val in properties.items():
            self.set_property(key, val, element=element)

    def get_elements_property(self, name):
        """Return a single property."""
        if name == "position":
            return np.array(self._positions)
        if name in self._el_props:
            return [i for i in self._el_props[name]]
        if name in self._grp_props:
            return [self._grp_props[name] for _ in range(len(self))]
        raise KeyError(f"{name} not in properties")

    def __len__(self):
        """Return the number of elements."""
        return len(self._coordinates)

    def __getitem__(self, index):
        """Return a single element."""
        try:
            index = int(index)
        except ValueError:
            raise TypeError(f"index must be an integer: {index}")
        return Element(
            name=self.name,
            type=self.etype,
            position=self._positions[index],
            **dict(
                [(str(k), v[index]) for k, v in self._el_props.items()]
                + [(str(k), v) for k, v in self._grp_props.items()]
            ),
        )

    def __iter__(self):
        """Iterate over elements."""
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        """Return representation string."""
        return (
            f"{self.__class__.__name__}(name={self.name}, elements={len(self)}, "
            f"el_properties=({', '.join(self._el_props.keys())}), "
            f"grp_properties=({', '.join(self._grp_props.keys())}))"
        )

    def unstack_coordinates(self):
        """Return a list of all coordinates in the group."""
        raise NotImplementedError

    def unstack_positions(self):
        """Return a list of all coordinates in the group."""
        raise NotImplementedError

    def update_positions(self, axes, offset, **kwargs):
        """Update element positions, give a axes basis and centre offset."""
        raise NotImplementedError

    def get_max_zposition(self):
        """Return the maximum z-coordinate."""
        raise NotImplementedError


class DrawElementsSphere(DrawElementsBase):
    """Store a set of 3D-visualisation sphere elements."""

    etype = "sphere"
    _protected_keys = ("name", "type", "position", "sradius", "get")

    def __init__(
        self,
        name,
        coordinates,
        radii,
        element_properties=None,
        group_properties=None,
        radii_scale=1.0,
    ):
        """Initialise the element group."""
        coordinates = np.array(coordinates)
        if coordinates.shape == (0,):
            coordinates = np.empty((0, 3))
        shape = coordinates.shape
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError(f"coordinates must be of the shape (N, 3) not {shape}")
        super().__init__(name, coordinates, element_properties, group_properties)
        self._radii = np.array(radii)
        self._radii_scale = radii_scale

    def __getitem__(self, index):
        """Return a single element."""
        try:
            index = int(index)
        except ValueError:
            raise TypeError(f"index must be an integer: {index}")
        return Element(
            name=self.name,
            type=self.etype,
            position=self._positions[index],
            sradius=self.scaled_radii[index],
            **dict(
                [(str(k), v[index]) for k, v in self._el_props.items()]
                + [(str(k), v) for k, v in self._grp_props.items()]
            ),
        )

    @property
    def scaled_radii(self):
        """Return the scaled radii, for each sphere."""
        return self._radii * self._radii_scale

    def unstack_coordinates(self):
        """Return a list of all coordinates in the group."""
        return self._coordinates

    def unstack_positions(self):
        """Return a list of all coordinates in the group."""
        return self._positions

    def update_positions(self, axes, offset, radii_scale, **kwargs):
        """Update element positions, give a axes basis and centre offset."""
        self._positions = np.dot(self._coordinates, axes) - offset
        self._axes = axes
        self._offset = offset
        self._radii_scale = radii_scale

    def get_max_zposition(self):
        """Return the maximum z-coordinate."""
        return self._positions[:, 2] + self.scaled_radii


class DrawElementsLine(DrawElementsBase):
    """Store a set of 3D-visualisation line elements."""

    etype = "line"

    def __init__(
        self, name, coordinates, element_properties=None, group_properties=None
    ):
        """Initialise the element group."""
        coordinates = np.array(coordinates)
        if coordinates.shape == (0,):
            coordinates = np.empty((0, 2, 3))
        shape = coordinates.shape
        if len(shape) != 3 or shape[1] != 2 or shape[2] != 3:
            raise ValueError(f"coordinates must be of the shape (N, 2, 3) not {shape}")
        super().__init__(name, coordinates, element_properties, group_properties)

    def unstack_coordinates(self):
        """Return a list of all coordinates in the group."""
        return np.concatenate((self._coordinates[:, 0, :], self._coordinates[:, 1, :]))

    def unstack_positions(self):
        """Return a list of all coordinates in the group."""
        return np.concatenate((self._positions[:, 0, :], self._positions[:, 1, :]))

    def update_positions(self, axes, offset, **kwargs):
        """Update element positions, give a axes basis and centre offset."""
        self._positions = np.einsum("ijk, km -> ijm", self._coordinates, axes) - offset
        self._axes = axes
        self._offset = offset

    def get_max_zposition(self):
        """Return the maximum z-coordinate."""
        return self._positions.max(axis=1)[:, 2]


class DrawElementsPoly(DrawElementsBase):
    """Store a set of 3D-visualisation polygon elements."""

    etype = "poly"

    # TODO validate init coordinate shapes

    def unstack_coordinates(self):
        """Return a list of all coordinates in the group."""
        planes = [np.array(plane) for plane in self._coordinates]
        if not planes:
            return np.empty((0, 3))
        return np.concatenate(planes)

    def unstack_positions(self):
        """Return a list of all coordinates in the group."""
        planes = [np.array(plane) for plane in self._positions]
        if not planes:
            return np.empty((0, 3))
        return np.concatenate(planes)

    def update_positions(self, axes, offset, **kwargs):
        """Update element positions, give a axes basis and centre offset."""
        # TODO ideally would apply transform to all planes at once
        self._positions = [np.dot(plane, axes) - offset for plane in self._coordinates]
        self._axes = axes
        self._offset = offset

    def get_max_zposition(self):
        """Return the maximum z-coordinate."""
        return np.array([plane[:, 2].max() for plane in self._positions])


class DrawGroup(Mapping):
    """Store and manipulate 3-D visualisation element groups."""

    def __init__(self, elements: List[DrawElementsBase]):
        """Store and manipulate 3-D visualisation element groups."""
        self._elements = OrderedDict([(el.name, el) for el in elements])

    def __getitem__(self, key):
        """Return an element group by name."""
        return self._elements[key]

    def __iter__(self):
        """Iterate over the element group names."""
        for key in self._elements:
            yield key

    def __len__(self):
        """Return the number of element groups."""
        return len(self._elements)

    def __repr__(self) -> str:
        """Return representation string."""
        return (
            f"{self.__class__.__name__}(groups=({', '.join(str(n) for n in self)}), "
            f"elements=({', '.join(str(len(self[n])) for n in self)}))"
        )

    def get_all_coordinates(self):
        """Return a list of all coordinates."""
        coordinates = [el.unstack_coordinates() for el in self._elements.values()]
        return np.concatenate(coordinates)

    def get_all_positions(self):
        """Return a list of all coordinates."""
        positions = [el.unstack_positions() for el in self._elements.values()]
        return np.concatenate(positions)

    def update_positions(self, axes=None, offset=None, radii_scale=1):
        """Update element positions, give a axes basis and centre offset."""
        if axes is None:
            axes = np.identity(3)
        if offset is None:
            offset = np.zeros(3)
        for element in self._elements.values():
            element.update_positions(axes, offset, radii_scale=radii_scale)

    def get_position_range(self):
        """Return the (minimum, maximum) coordinates."""
        min_positions = []
        max_positions = []
        for element in self._elements.values():
            positions = element.unstack_positions()
            if isinstance(element, DrawElementsSphere):  # type: DrawElementsSphere
                # TODO make more general
                min_positions.append(positions - element.scaled_radii[:, None])
                max_positions.append(positions + element.scaled_radii[:, None])
            else:
                min_positions.append(positions)
                max_positions.append(positions)
        return (
            np.concatenate(min_positions).min(0),
            np.concatenate(max_positions).max(0),
        )

    def yield_zorder(self):
        """Yield elements, in order of the z-coordinate."""
        keys = [(el.name, i) for el in self.values() for i in range(len(el))]
        z_positions = np.concatenate([el.get_max_zposition() for el in self.values()])
        for i in z_positions.argsort():
            yield i, self[keys[i][0]][keys[i][1]]
