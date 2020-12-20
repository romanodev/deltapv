from jaxpv import objects, physics, current, util
from typing import Tuple

PVCell = objects.PVCell
LightSource = objects.LightSource
Array = util.Array
f64 = util.f64


def contact_phin(cell: PVCell, neq0: f64, neqL: f64, phi_n: Array,
                 phi: Array) -> f64:

    n = physics.n(cell, phi_n, phi)
    Jn = current.Jn(cell, phi_n, phi)
    return Jn[0] - cell.Snl * (n[0] - neq0), Jn[-1] + cell.Snr * (n[-1] - neqL)


def contact_phin_deriv(cell: PVCell, phi_n: Array,
                       phi: Array) -> Tuple[f64, f64, f64, f64]:

    n = physics.n(cell, phi_n, phi)
    dJn_phin_maindiag, dJn_phin_upperdiag, dJn_phi_maindiag, dJn_phi_upperdiag = current.Jn_deriv(
        cell, phi_n, phi)

    return dJn_phin_maindiag[0] - cell.Snl * n[0] , dJn_phin_upperdiag[0] , \
    dJn_phi_maindiag[0] - cell.Snl * n[0] , dJn_phi_upperdiag[0] , \
    dJn_phin_maindiag[-1] , dJn_phin_upperdiag[-1] + cell.Snr * n[-1] , \
    dJn_phi_maindiag[-1] , dJn_phi_upperdiag[-1] + cell.Snr * n[-1]


def contact_phip(cell: PVCell, peq0: f64, peqL: f64, phi_p: Array,
                 phi: Array) -> f64:

    p = physics.p(cell, phi_p, phi)
    Jp = current.Jp(cell, phi_p, phi)
    return Jp[0] + cell.Spl * (p[0] - peq0), Jp[-1] - cell.Spr * (p[-1] - peqL)


def contact_phip_deriv(cell: PVCell, phi_p: Array,
                       phi: Array) -> Tuple[f64, f64, f64, f64]:

    p = physics.p(cell, phi_p, phi)
    dJp_phip_maindiag, dJp_phip_upperdiag, dJp_phi_maindiag, dJp_phi_upperdiag = current.Jp_deriv(
        cell, phi_p, phi)

    return dJp_phip_maindiag[0] - cell.Spl * p[0] , dJp_phip_upperdiag[0] , \
    dJp_phi_maindiag[0] - cell.Spl * p[0] , dJp_phi_upperdiag[0] , \
    dJp_phip_maindiag[-1] , dJp_phip_upperdiag[-1] + cell.Spr * p[-1] , \
    dJp_phi_maindiag[-1] , dJp_phi_upperdiag[-1] + cell.Spr * p[-1]
