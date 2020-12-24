from jaxpv import objects, physics, current, util
from typing import Tuple

PVCell = objects.PVCell
Potentials = objects.Potentials
Boundary = objects.Boundary
Array = util.Array
f64 = util.f64


def contact_phin(cell: PVCell, bound: Boundary, pot: Potentials) -> f64:

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


def contact_phip(cell: PVCell, bound: Boundary, pot: Potentials) -> f64:

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
