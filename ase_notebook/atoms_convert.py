"""module for serializing ``ase.Atoms``.

Note: there is a json serializer in ``ase.io.jsonio``,
but it doesn't actually store all the data
"""
import datetime
import json

import ase
from ase.constraints import dict2constraint
import numpy as np


class ASEEncoder(json.JSONEncoder):
    """JSON Encoder for ase.Atoms serialization."""

    def default(self, obj):
        """Parse object."""
        if hasattr(obj, "todict"):
            d = obj.todict()

            if not isinstance(d, dict):
                raise RuntimeError(
                    f"todict() of {obj} returned object of type {type(d)} "
                    "but should have returned dict"
                )
            if hasattr(obj, "ase_objtype"):
                d["__ase_objtype__"] = obj.ase_objtype

            return d
        if isinstance(obj, np.ndarray) or hasattr(obj, "__array__"):
            if obj.dtype == complex:
                return {"__complex_ndarray__": (obj.real.tolist(), obj.imag.tolist())}
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, datetime.datetime):
            return {"__datetime__": obj.isoformat()}
        return json.JSONEncoder.default(self, obj)


def try_int(obj):
    """Try conversion of object to int."""
    try:
        return int(obj)
    except ValueError:
        return obj


def numpyfy(obj):
    """Convert an object to numpy array(s) recursively."""
    if isinstance(obj, dict):
        if "__complex_ndarray__" in obj:
            r, i = (np.array(x) for x in obj["__complex_ndarray__"])
            return r + i * 1j
        return {try_int(key): numpyfy(value) for key, value in obj.items()}
    if isinstance(obj, list) and len(obj) > 0:
        try:
            a = np.array(obj)
        except ValueError:
            pass
        else:
            if a.dtype in [bool, int, float]:
                return a
        obj = [numpyfy(value) for value in obj]
    return obj


def ase_decoder_hook(dct):
    """JSON decoder hook for ase.Atoms de-serialization."""
    if "__datetime__" in dct:
        return datetime.datetime.strptime(dct["__datetime__"], "%Y-%m-%dT%H:%M:%S.%f")
    if "__complex_ndarray__" in dct:
        r, i = (np.array(x) for x in dct["__complex_ndarray__"])
        return r + i * 1j

    if "__ase_objtype__" in dct:
        objtype = dct.pop("__ase_objtype__")
        dct = numpyfy(dct)

        if objtype == "cell":
            from ase.cell import Cell

            obj = Cell(**dct)
        else:
            raise RuntimeError(
                "Do not know how to decode object type {} "
                "into an actual object".format(objtype)
            )

        assert obj.ase_objtype == objtype
        return obj

    return dct


def serialize_atoms(atoms: ase.Atoms, description: str = "") -> dict:
    """Serialize an ase.Atoms instance to a dictionary."""
    dct = {
        "description": description,
        "cell": atoms.cell,
        "arrays": atoms.arrays,
        "info": atoms.info,
        "constraints": atoms.constraints,
        "celldisp": atoms.get_celldisp(),
        "calculator": atoms.calc,
    }
    return ASEEncoder().encode(dct)


def deserialize_atoms(dct: dict) -> ase.Atoms:
    """Deserialize a dictionary to an ase.Atoms instance."""
    dct = json.JSONDecoder(object_hook=ase_decoder_hook).decode(dct)
    atoms = ase.Atoms()
    atoms.cell = dct["cell"]
    atoms.arrays = numpyfy(dct["arrays"])
    atoms.info = dct["info"]
    atoms.constraints = [dict2constraint(d) for d in dct["constraints"]]
    atoms.set_celldisp(dct["celldisp"])
    # TODO ase.calculators.calculator.Calculator has a todict method,
    # but not clear how to convert it back

    return atoms


def convert_to_atoms(obj):
    """Attempt to convert an object to an ase.Atoms object."""
    if isinstance(obj, ase.Atoms):
        return obj

    if isinstance(obj, dict):
        return deserialize_atoms(obj)

    if hasattr(obj, "lattice") and hasattr(obj, "sites"):
        # we assume the obj is a pymatgen Structure

        # from pymatgen.io.ase adaptor
        if not obj.is_ordered:
            raise ValueError("ASE Atoms only supports ordered Pymatgen structures")
        symbols = [str(site.specie.symbol) for site in obj]
        positions = [site.coords for site in obj]
        cell = obj.lattice.matrix
        # TODO test if slab, then use pbc = [True, True, False]
        atoms = ase.Atoms(symbols=symbols, positions=positions, pbc=True, cell=cell)

        # additionally, propagate site properties
        for key, array in obj.site_properties.items():
            if key not in atoms.arrays:
                atoms.set_array(key, np.array(array))
        # TODO propagate partial occupancies, and other properties

        return atoms

    raise TypeError(f"Cannot convert object of type {obj.__class__.__name__}")
