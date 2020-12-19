from jaxpv import objects, physics, util
from jax import numpy as np
from typing import Tuple

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def comp_rad(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Array:

    ni = physics.ni(cell)
    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)
    return cell.Br * (n * p - ni**2)


def comp_rad_deriv(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Tuple[Array, Array, Array]:

    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)

    DR_phin = cell.Br * (n * p)
    DR_phip = cell.Br * (-n * p)
    DR_phi = np.zeros_like(phi)

    return DR_phin, DR_phip, DR_phi
