from jaxpv import objects, recombination as rec, current, util
from jax import numpy as np
from typing import Tuple

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def ddp(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Array:

    R = rec.comp_SHR(cell, phi_n, phi_p, phi) + rec.comp_auger(
        cell, phi_n, phi_p, phi)
    Jp = current.Jp(cell, phi_p, phi)
    ave_dgrid = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2.
    return np.diff(Jp) / ave_dgrid + R[1:-1] - cell.G[1:-1]


def ddp_deriv(
        cell: PVCell, phi_n: Array, phi_p: Array,
        phi: Array) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:

    DR_SHR_phin, DR_SHR_phip, DR_SHR_phi = rec.comp_SHR_deriv(
        cell, phi_n, phi_p, phi)
    DR_rad_phin, DR_rad_phip, DR_rad_phi = rec.comp_rad_deriv(
        cell, phi_n, phi_p, phi)
    DR_auger_phin, DR_auger_phip, DR_auger_phi = rec.comp_auger_deriv(
        cell, phi_n, phi_p, phi)

    DR_phin = DR_SHR_phin + DR_rad_phin + DR_auger_phin
    DR_phip = DR_SHR_phip + DR_rad_phip + DR_auger_phip
    DR_phi = DR_SHR_phi + DR_rad_phi + DR_auger_phi

    dJp_phip_maindiag, dJp_phip_upperdiag, dJp_phi_maindiag, dJp_phi_upperdiag = current.Jp_deriv(
        cell, phi_p, phi)

    ave_dgrid = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2.

    ddp_phip_ = -dJp_phip_maindiag[:-1] / ave_dgrid
    ddp_phip__ = (-dJp_phip_upperdiag[:-1] +
                  dJp_phip_maindiag[1:]) / ave_dgrid + DR_phip[1:-1]
    ddp_phip___ = dJp_phip_upperdiag[1:] / ave_dgrid

    ddp_phi_ = -dJp_phi_maindiag[:-1] / ave_dgrid
    ddp_phi__ = (-dJp_phi_upperdiag[:-1] +
                 dJp_phi_maindiag[1:]) / ave_dgrid + DR_phi[1:-1]
    ddp_phi___ = dJp_phi_upperdiag[1:] / ave_dgrid

    ddp_phin__ = DR_phin[1:-1]

    return ddp_phin__, ddp_phip_, ddp_phip__, ddp_phip___, ddp_phi_, ddp_phi__, ddp_phi___


def ddn(cell: PVCell, phi_n: Array, phi_p: Array, phi: Array) -> Array:

    R = rec.comp_SHR(cell, phi_n, phi_p, phi) \
        + rec.comp_rad(cell, phi_n, phi_p, phi) \
        + rec.comp_auger(cell, phi_n, phi_p, phi)

    Jn = current.Jn(cell, phi_n, phi)

    ave_dgrid = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2.

    return np.diff(Jn) / ave_dgrid - R[1:-1] + cell.G[1:-1]


def ddn_deriv(
        cell: PVCell, phi_n: Array, phi_p: Array,
        phi: Array) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:

    DR_SHR_phin, DR_SHR_phip, DR_SHR_phi = rec.comp_SHR_deriv(
        cell, phi_n, phi_p, phi)
    DR_rad_phin, DR_rad_phip, DR_rad_phi = rec.comp_rad_deriv(
        cell, phi_n, phi_p, phi)
    DR_auger_phin, DR_auger_phip, DR_auger_phi = rec.comp_auger_deriv(
        cell, phi_n, phi_p, phi)

    DR_phin = DR_SHR_phin + DR_rad_phin + DR_auger_phin
    DR_phip = DR_SHR_phip + DR_rad_phip + DR_auger_phip
    DR_phi = DR_SHR_phi + DR_rad_phi + DR_auger_phi

    dJn_phin_maindiag, dJn_phin_upperdiag, dJn_phi_maindiag, dJn_phi_upperdiag = current.Jn_deriv(
        cell, phi_n, phi)

    ave_dgrid = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2.

    dde_phin_ = -dJn_phin_maindiag[:-1] / ave_dgrid
    dde_phin__ = (-dJn_phin_upperdiag[:-1] +
                  dJn_phin_maindiag[1:]) / ave_dgrid - DR_phin[1:-1]
    dde_phin___ = dJn_phin_upperdiag[1:] / ave_dgrid

    dde_phi_ = -dJn_phi_maindiag[:-1] / ave_dgrid
    dde_phi__ = (-dJn_phi_upperdiag[:-1] +
                 dJn_phi_maindiag[1:]) / ave_dgrid - DR_phi[1:-1]
    dde_phi___ = dJn_phi_upperdiag[1:] / ave_dgrid

    dde_phip__ = -DR_phip[1:-1]

    return dde_phin_, dde_phin__, dde_phin___, dde_phip__, dde_phi_, dde_phi__, dde_phi___
