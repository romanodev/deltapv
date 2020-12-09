from . import SHR
from . import auger
from . import radiative
from . import current

import jax.numpy as np


def ddn(data, phi_n, phi_p, phi):

    dgrid = data["dgrid"]
    G = data["G"]
    R = SHR.comp_SHR(data, phi_n, phi_p, phi) \
        + radiative.comp_rad(data, phi_n, phi_p, phi) \
        + auger.comp_auger(data, phi_n, phi_p, phi)

    Jn = current.Jn(data, phi_n, phi)

    ave_dgrid = (dgrid[:-1] + dgrid[1:]) / 2

    return np.diff(Jn) / ave_dgrid - R[1:-1] + G[1:-1]


def ddn_deriv(data, phi_n, phi_p, phi):

    dgrid = data["dgrid"]
    G = data["G"]
    DR_SHR_phin, DR_SHR_phip, DR_SHR_phi = SHR.comp_SHR_deriv(
        data, phi_n, phi_p, phi)
    DR_rad_phin, DR_rad_phip, DR_rad_phi = radiative.comp_rad_deriv(
        data, phi_n, phi_p, phi)
    DR_auger_phin, DR_auger_phip, DR_auger_phi = auger.comp_auger_deriv(
        data, phi_n, phi_p, phi)

    DR_phin = DR_SHR_phin + DR_rad_phin + DR_auger_phin
    DR_phip = DR_SHR_phip + DR_rad_phip + DR_auger_phip
    DR_phi = DR_SHR_phi + DR_rad_phi + DR_auger_phi

    dJn_phin_maindiag, dJn_phin_upperdiag, dJn_phi_maindiag, dJn_phi_upperdiag = current.Jn_deriv(
        data, phi_n, phi)

    ave_dgrid = (dgrid[:-1] + dgrid[1:]) / 2.0

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
