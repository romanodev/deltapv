from . import IV
from . import scaling

import jax.numpy as np

scale = scaling.scales()


def Vincrement(data, num_vals=50):

    Chi, Eg, Nc, Nv, Ndop = data["Chi"], data["Eg"], data["Nc"], data[
        "Nv"], data["Ndop"]

    if Ndop[0] > 0:
        phi_ini_left = -Chi[0] + np.log((Ndop[0]) / Nc[0])
    else:
        phi_ini_left = -Chi[0] - Eg[0] - np.log(np.abs(Ndop[0]) / Nv[0])

    if Ndop[-1] > 0:
        phi_ini_right = -Chi[-1] + np.log((Ndop[-1]) / Nc[-1])
    else:
        phi_ini_right = -Chi[-1] - Eg[-1] - np.log(np.abs(Ndop[-1]) / Nv[-1])

    incr_step = np.abs(phi_ini_right - phi_ini_left) / num_vals
    incr_sign = -1 if phi_ini_right > phi_ini_left else 1

    return incr_sign * incr_step


def compute_eff(data, Vincrement):

    current = IV.calc_IV(data, Vincrement)

    voltages = np.linspace(start=0,
                           stop=(current.size - 1) * Vincrement,
                           num=current.size)

    Pmax = np.max(scale['E'] * voltages * scale['J'] * current) * 1e4  # W/m2

    eff = Pmax / 1e3  # P_in is normalized: np.sum( P_in ) = 1000 W/m2 = 1 sun

    return eff
