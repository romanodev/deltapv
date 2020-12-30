from jaxpv import objects, physics, current, util
from jax import numpy as np
from typing import Tuple

PVCell = objects.PVCell
Potentials = objects.Potentials
Boundary = objects.Boundary
Array = util.Array
f64 = util.f64


def boundary_phi(cell: PVCell) -> Tuple[f64, f64]:

    phi0 = np.where(
        cell.Ndop[0] > 0,
        -cell.Chi[0] + np.log(np.abs(cell.Ndop[0] / cell.Nc[0])),
        -cell.Chi[0] - cell.Eg[0] - np.log(np.abs(-cell.Ndop[0] / cell.Nv[0])))

    phiL = np.where(
        cell.Ndop[-1] > 0,
        -cell.Chi[-1] + np.log(np.abs(cell.Ndop[-1] / cell.Nc[-1])),
        -cell.Chi[-1] - cell.Eg[-1] -
        np.log(np.abs(-cell.Ndop[-1] / cell.Nv[-1])))

    return phi0, phiL


def boundary_eq(cell: PVCell) -> Boundary:

    phi0, phiL = boundary_phi(cell)

    return Boundary(phi0, phiL, f64(0), f64(0), f64(0), f64(0))


def boundary(cell: PVCell, v: f64) -> Boundary:

    phi0, phiLeq = boundary_phi(cell)
    neq0 = cell.Nc[0] * np.exp(cell.Chi[0] + phi0)
    neqL = cell.Nc[-1] * np.exp(cell.Chi[-1] + phiLeq)
    peq0 = cell.Nv[0] * np.exp(-cell.Chi[0] - cell.Eg[0] - phi0)
    peqL = cell.Nv[-1] * np.exp(-cell.Chi[-1] - cell.Eg[-1] - phiLeq)

    return Boundary(phi0, phiLeq + v, neq0, neqL, peq0, peqL)


def contact_phin(cell: PVCell, bound: Boundary,
                 pot: Potentials) -> Tuple[f64, f64]:

    n = physics.n(cell, pot)
    Jn = current.Jn(cell, pot)
    return Jn[0] - cell.Snl * (n[0] - bound.neq0), Jn[-1] + cell.Snr * (
        n[-1] - bound.neqL)


def contact_phin_deriv(cell: PVCell,
                       pot: Potentials) -> Tuple[f64, f64, f64, f64]:

    n = physics.n(cell, pot)
    dJn_phin_maindiag, dJn_phin_upperdiag, dJn_phi_maindiag, dJn_phi_upperdiag = current.Jn_deriv(
        cell, pot)

    return dJn_phin_maindiag[0] - cell.Snl * n[0] , dJn_phin_upperdiag[0] , \
    dJn_phi_maindiag[0] - cell.Snl * n[0] , dJn_phi_upperdiag[0] , \
    dJn_phin_maindiag[-1] , dJn_phin_upperdiag[-1] + cell.Snr * n[-1] , \
    dJn_phi_maindiag[-1] , dJn_phi_upperdiag[-1] + cell.Snr * n[-1]


def contact_phip(cell: PVCell, bound: Boundary,
                 pot: Potentials) -> Tuple[f64, f64]:

    p = physics.p(cell, pot)
    Jp = current.Jp(cell, pot)
    return Jp[0] + cell.Spl * (p[0] - bound.peq0), Jp[-1] - cell.Spr * (
        p[-1] - bound.peqL)


def contact_phip_deriv(cell: PVCell,
                       pot: Potentials) -> Tuple[f64, f64, f64, f64]:

    p = physics.p(cell, pot)
    dJp_phip_maindiag, dJp_phip_upperdiag, dJp_phi_maindiag, dJp_phi_upperdiag = current.Jp_deriv(
        cell, pot)

    return dJp_phip_maindiag[0] - cell.Spl * p[0] , dJp_phip_upperdiag[0] , \
    dJp_phi_maindiag[0] - cell.Spl * p[0] , dJp_phi_upperdiag[0] , \
    dJp_phip_maindiag[-1] , dJp_phip_upperdiag[-1] + cell.Spr * p[-1] , \
    dJp_phi_maindiag[-1] , dJp_phi_upperdiag[-1] + cell.Spr * p[-1]


def contact_phi(cell: PVCell, bound: Boundary,
                pot: Potentials) -> Tuple[f64, f64]:

    return pot.phi[0] - bound.phi0, pot.phi[-1] - bound.phiL
