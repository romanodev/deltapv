from . import SHR
from . import auger
from . import radiative
from . import current

import jax.numpy as np


def ddp(data, phi_n, phi_p, phi):

    dgrid = data["dgrid"]
    G = data["G"]
    R = SHR.comp_SHR(data, phi_n, phi_p, phi) + auger.comp_auger(
        data, phi_n, phi_p, phi)
    Jp = current.Jp(data, phi_p, phi)
    ave_dgrid = (dgrid[:-1] + dgrid[1:]) / 2.0
    return np.diff(Jp) / ave_dgrid + R[1:-1] - G[1:-1]


def ddp_deriv(data, phi_n, phi_p, phi):

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

    dJp_phip_maindiag, dJp_phip_upperdiag, dJp_phi_maindiag, dJp_phi_upperdiag = current.Jp_deriv(
        data, phi_p, phi)

    ave_dgrid = (dgrid[:-1] + dgrid[1:]) / 2.0

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
