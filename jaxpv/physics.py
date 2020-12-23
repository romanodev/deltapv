from jaxpv import objects, util
from jax import numpy as np

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def n(cell: PVCell, phi_n: Array, phi: Array) -> Array:

    return cell.Nc * np.exp(cell.Chi + phi_n + phi)


def p(cell: PVCell, phi_p: Array, phi: Array) -> Array:

    return cell.Nv * np.exp(-cell.Chi - cell.Eg - phi_p - phi)


def charge(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Array:

    _n = n(cell, phi_n, phi)
    _p = p(cell, phi_p, phi)
    return -_n + _p + cell.Ndop


def ni(cell: PVCell) -> Array:

    return np.sqrt(cell.Nc * cell.Nv) * np.exp(-cell.Eg / 2)
