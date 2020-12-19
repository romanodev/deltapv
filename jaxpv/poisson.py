from jaxpv import objects, physics, util
from jax import numpy as np
from typing import Tuple

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def pois(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Array:

    ave_dgrid = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2.
    ave_eps = (cell.eps[1:] + cell.eps[:-1]) / 2.
    pois = (ave_eps[:-1] * np.diff(phi)[:-1] / cell.dgrid[:-1] - ave_eps[1:] *
            np.diff(phi)[1:] / cell.dgrid[1:]) / ave_dgrid - physics.charge(
                cell, phi_n, phi_p, phi)[1:-1]
    return pois


def pois_deriv_eq(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Tuple[Array, Array, Array]:

    ave_dgrid = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2.
    ave_eps = (cell.eps[1:] + cell.eps[:-1]) / 2.
    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)

    dchg_phi = -n - p

    dpois_phi_ = -ave_eps[:-1] / cell.dgrid[:-1] / ave_dgrid
    dpois_phi__ = (ave_eps[:-1] / cell.dgrid[:-1] +
                   ave_eps[1:] / cell.dgrid[1:]) / ave_dgrid - dchg_phi[1:-1]
    dpois_phi___ = -ave_eps[1:] / cell.dgrid[1:] / ave_dgrid

    return dpois_phi_, dpois_phi__, dpois_phi___


def pois_deriv(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Tuple[Array, Array, Array, Array, Array]:

    ave_dgrid = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2.0
    ave_eps = 0.5 * (cell.eps[1:] + cell.eps[:-1])
    n = physics.n(cell, phi_n, phi)
    p = physics.p(cell, phi_p, phi)

    dchg_phi_n = -n
    dchg_phi_p = -p
    dchg_phi = -n - p

    dpois_phi_ = -ave_eps[:-1] / cell.dgrid[:-1] / ave_dgrid
    dpois_phi__ = (ave_eps[:-1] / cell.dgrid[:-1] +
                   ave_eps[1:] / cell.dgrid[1:]) / ave_dgrid - dchg_phi[1:-1]
    dpois_phi___ = -ave_eps[1:] / cell.dgrid[1:] / ave_dgrid

    dpois_dphin__ = -dchg_phi_n[1:-1]
    dpois_dphip__ = -dchg_phi_p[1:-1]

    return dpois_phi_, dpois_phi__, dpois_phi___, dpois_dphin__, dpois_dphip__
