from jaxpv import objects, util
from jax import numpy as np

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


def n(cell: PVCell, pot: Potentials) -> Array:

    return cell.Nc * np.exp(cell.Chi + pot.phi_n + pot.phi)


def p(cell: PVCell, pot: Potentials) -> Array:

    return cell.Nv * np.exp(-cell.Chi - cell.Eg - pot.phi_p - pot.phi)


def charge(cell: PVCell, pot: Potentials) -> Array:

    _n = n(cell, pot)
    _p = p(cell, pot)
    return -_n + _p + cell.Ndop


def ni(cell: PVCell) -> Array:

    return np.sqrt(cell.Nc * cell.Nv) * np.exp(-cell.Eg / 2)
