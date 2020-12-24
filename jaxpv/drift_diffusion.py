from jaxpv import objects, recombination as rec, current, util
from jax import numpy as np
from typing import Tuple

PVCell = objects.PVCell
LightSource = objects.LightSource
Potentials = objects.Potentials
Array = util.Array
f64 = util.f64


def ddp(cell: PVCell, pot: Potentials) -> Array:

    R = rec.all_recomb(cell, pot)
    Jp = current.Jp(cell, pot)
    ave_dgrid = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2.
    return R[1:-1] - cell.G[1:-1] + np.diff(Jp) / ave_dgrid


def ddp_deriv(
        cell: PVCell, pot: Potentials
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:

    DR_phin, DR_phip, DR_phi = rec.all_recomb_deriv(cell, pot)

    dJp_phip_maindiag, dJp_phip_upperdiag, dJp_phi_maindiag, dJp_phi_upperdiag = current.Jp_deriv(
        cell, pot)

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


def ddn(cell: PVCell, pot: Potentials) -> Array:

    R = rec.all_recomb(cell, pot)

    Jn = current.Jn(cell, pot)

    ave_dgrid = (cell.dgrid[:-1] + cell.dgrid[1:]) / 2.

    return -R[1:-1] + cell.G[1:-1] + np.diff(Jn) / ave_dgrid


def ddn_deriv(
        cell: PVCell, pot: Potentials
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:

    DR_phin, DR_phip, DR_phi = rec.all_recomb_deriv(cell, pot)

    dJn_phin_maindiag, dJn_phin_upperdiag, dJn_phi_maindiag, dJn_phi_upperdiag = current.Jn_deriv(
        cell, pot)

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
