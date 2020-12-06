from . import physics

import jax.numpy as np


def pois(data, phi_n, phi_p, phi):
    
    dgrid = data["dgrid"]
    eps = data["eps"]
    ave_dgrid = (dgrid[:-1] + dgrid[1:]) / 2.0
    ave_eps = 0.5 * (eps[1:] + eps[:-1])
    pois = (ave_eps[:-1] * np.diff(phi)[:-1] / dgrid[:-1] - ave_eps[1:] *
            np.diff(phi)[1:] / dgrid[1:]) / ave_dgrid - physics.charge(
                data, phi_n, phi_p, phi)[1:-1]
    return pois


def pois_deriv_eq(data, phi_n, phi_p, phi):
    
    dgrid = data["dgrid"]
    eps = data["eps"]
    ave_dgrid = (dgrid[:-1] + dgrid[1:]) / 2.0
    ave_eps = 0.5 * (eps[1:] + eps[:-1])
    n = physics.n(data, phi_n, phi)
    p = physics.p(data, phi_p, phi)

    dchg_phi = -n - p

    dpois_phi_ = -ave_eps[:-1] / dgrid[:-1] / ave_dgrid
    dpois_phi__ = (ave_eps[:-1] / dgrid[:-1] +
                   ave_eps[1:] / dgrid[1:]) / ave_dgrid - dchg_phi[1:-1]
    dpois_phi___ = -ave_eps[1:] / dgrid[1:] / ave_dgrid

    return dpois_phi_, dpois_phi__, dpois_phi___


def pois_deriv(data, phi_n, phi_p, phi):
    
    dgrid = data["dgrid"]
    eps = data["eps"]
    ave_dgrid = (dgrid[:-1] + dgrid[1:]) / 2.0
    ave_eps = 0.5 * (eps[1:] + eps[:-1])
    n = physics.n(data, phi_n, phi)
    p = physics.p(data, phi_p, phi)

    dchg_phi_n = -n
    dchg_phi_p = -p
    dchg_phi = -n - p

    dpois_phi_ = -ave_eps[:-1] / dgrid[:-1] / ave_dgrid
    dpois_phi__ = (ave_eps[:-1] / dgrid[:-1] +
                   ave_eps[1:] / dgrid[1:]) / ave_dgrid - dchg_phi[1:-1]
    dpois_phi___ = -ave_eps[1:] / dgrid[1:] / ave_dgrid

    dpois_dphin__ = -dchg_phi_n[1:-1]
    dpois_dphip__ = -dchg_phi_p[1:-1]

    return dpois_phi_, dpois_phi__, dpois_phi___, dpois_dphin__, dpois_dphip__
