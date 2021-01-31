from deltapv import objects, physics, util
from jax import numpy as np
from typing import Tuple

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


def all_recomb(cell: PVCell, pot: Potentials) -> Array:

    return comp_auger(cell, pot) + comp_shr(cell, pot) + comp_rad(cell, pot)


def all_recomb_deriv(cell: PVCell,
                     pot: Potentials) -> Tuple[Array, Array, Array]:

    auger_phi_n, auger_phi_p, auger_phi = comp_auger_deriv(cell, pot)
    shr_phi_n, shr_phi_p, shr_phi = comp_shr_deriv(cell, pot)
    rad_phi_n, rad_phi_p, rad_phi = comp_rad_deriv(cell, pot)

    DR_phin = auger_phi_n + shr_phi_n + rad_phi_n
    DR_phip = auger_phi_p + shr_phi_p + rad_phi_p
    DR_phi = auger_phi + shr_phi + rad_phi

    return DR_phin, DR_phip, DR_phi


def comp_auger(cell: PVCell, pot: Potentials) -> Array:

    ni = physics.ni(cell)
    n = physics.n(cell, pot)
    p = physics.p(cell, pot)
    return (cell.Cn * n + cell.Cp * p) * (n * p - ni**2)


def comp_auger_deriv(cell: PVCell,
                     pot: Potentials) -> Tuple[Array, Array, Array]:

    ni = physics.ni(cell)
    n = physics.n(cell, pot)
    p = physics.p(cell, pot)

    DR_phin = (cell.Cn * n) * (n * p - ni**2) + (cell.Cn * n +
                                                 cell.Cp * p) * (n * p)
    DR_phip = (-cell.Cp * p) * (n * p - ni**2) + (cell.Cn * n +
                                                  cell.Cp * p) * (-n * p)
    DR_phi = (cell.Cn * n - cell.Cp * p) * (n * p - ni**2)

    return DR_phin, DR_phip, DR_phi


def comp_shr(cell: PVCell, pot: Potentials) -> Array:

    ni = physics.ni(cell)
    n = physics.n(cell, pot)
    p = physics.p(cell, pot)
    nR = ni * np.exp(cell.Et) + n
    pR = ni * np.exp(-cell.Et) + p
    return (n * p - ni**2) / (cell.tp * nR + cell.tn * pR)


def comp_shr_deriv(cell: PVCell,
                   pot: Potentials) -> Tuple[Array, Array, Array]:

    ni = physics.ni(cell)
    n = physics.n(cell, pot)
    p = physics.p(cell, pot)
    nR = ni * np.exp(cell.Et) + n
    pR = ni * np.exp(-cell.Et) + p
    num = n * p - ni**2
    denom = cell.tp * nR + cell.tn * pR

    DR_phin = ((n * p) * denom - num * (cell.tp * n)) * denom**(-2)
    DR_phip = ((-n * p) * denom - num * (-cell.tn * p)) * denom**(-2)
    DR_phi = (-num * (cell.tp * n - cell.tn * p)) * denom**(-2)

    return DR_phin, DR_phip, DR_phi


def comp_rad(cell: PVCell, pot: Potentials) -> Array:

    ni = physics.ni(cell)
    n = physics.n(cell, pot)
    p = physics.p(cell, pot)
    return cell.Br * (n * p - ni**2)


def comp_rad_deriv(cell: PVCell,
                   pot: Potentials) -> Tuple[Array, Array, Array]:

    n = physics.n(cell, pot)
    p = physics.p(cell, pot)

    DR_phin = cell.Br * (n * p)
    DR_phip = cell.Br * (-n * p)
    DR_phi = np.zeros_like(pot.phi)

    return DR_phin, DR_phip, DR_phi
