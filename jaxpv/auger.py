from jaxpv import objects, physics, util
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


def comp_auger_deriv(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Tuple[Array, Array, Array]:

    ni = physics.ni(cell)
    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)

    DR_phin = (cell.Cn * n) * (n * p - ni**2) + (cell.Cn * n + cell.Cp * p) * (n * p)
    DR_phip = (-cell.Cp * p) * (n * p - ni**2) + (cell.Cn * n + cell.Cp * p) * (-n * p)
    DR_phi = (cell.Cn * n - cell.Cp * p) * (n * p - ni**2)

    return DR_phin, DR_phip, DR_phi
