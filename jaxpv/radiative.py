from . import physics

import jax.numpy as np


def comp_rad(data, phi_n, phi_p, phi):

    Br = data["Br"]
    ni = physics.ni(data)
    n = physics.n(data, phi_n, phi)
    p = physics.p(data, phi_p, phi)
    return Br * (n * p - ni**2)


def comp_rad_deriv(data, phi_n, phi_p, phi):

    Br = data["Br"]
    n = physics.n(data, phi_n, phi)
    p = physics.p(data, phi_p, phi)

    DR_phin = Br * (n * p)
    DR_phip = Br * (-n * p)
    DR_phi = np.zeros_like(phi)

    return DR_phin, DR_phip, DR_phi
