from jaxpv import objects, physics, util
from jax import numpy as np
from typing import Tuple

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def comp_auger(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Array:

    ni = physics.ni(cell)
    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)
    return (cell.Cn * n + cell.Cp * p) * (n * p - ni**2)


def comp_auger_deriv(cell: PVCell, phi_n: Array, phi_p: Array,
                     phi: Array) -> Tuple[Array, Array, Array]:

    ni = physics.ni(cell)
    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)

    DR_phin = (cell.Cn * n) * (n * p - ni**2) + (cell.Cn * n +
                                                 cell.Cp * p) * (n * p)
    DR_phip = (-cell.Cp * p) * (n * p - ni**2) + (cell.Cn * n +
                                                  cell.Cp * p) * (-n * p)
    DR_phi = (cell.Cn * n - cell.Cp * p) * (n * p - ni**2)

    return DR_phin, DR_phip, DR_phi


def comp_SHR(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Array:

    ni = physics.ni(cell)
    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)
    nR = ni * np.exp(cell.Et) + n
    pR = ni * np.exp(-cell.Et) + p
    return (n * p - ni**2) / (cell.tp * nR + cell.tn * pR)


def comp_SHR_deriv(cell: PVCell, phi_n: Array, phi_p: Array,
                   phi: Array) -> Tuple[Array, Array, Array]:

    ni = physics.ni(cell)
    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)
    nR = ni * np.exp(cell.Et) + n
    pR = ni * np.exp(-cell.Et) + p
    num = n * p - ni**2
    denom = cell.tp * nR + cell.tn * pR

    DR_phin = ((n * p) * denom - num * (cell.tp * n)) * denom**(-2)
    DR_phip = ((-n * p) * denom - num * (-cell.tn * p)) * denom**(-2)
    DR_phi = (-num * (cell.tp * n - cell.tn * p)) * denom**(-2)

    return DR_phin, DR_phip, DR_phi


def comp_rad(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Array:

    ni = physics.ni(cell)
    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)
    return cell.Br * (n * p - ni**2)


def comp_rad_deriv(cell: PVCell, phi_n: Array, phi_p: Array,
                   phi: Array) -> Tuple[Array, Array, Array]:

    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)

    DR_phin = cell.Br * (n * p)
    DR_phip = cell.Br * (-n * p)
    DR_phi = np.zeros_like(phi)

    return DR_phin, DR_phip, DR_phi
