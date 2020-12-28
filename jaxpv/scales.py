from scipy import constants as const
from jax import numpy as np

##### Units required in inputs #####
## grid : cm
## eps : unitless
## Chi , Eg , Et : eV
## Nc , Nv , Ndop : cm^(-3)
## mn , mp : cm^2 / V / s
## tn , tp : s
## Snl , Snr , Spl , Spr : cm / s
## G : cm^(-3) / s

T = 3e2
KB = const.k
hc = const.c * const.h * 1e9  # J.nm
_q, _, _ = const.physical_constants["elementary charge"]
_eps_0 = const.epsilon_0 * 1e-2  # C * V-1 * m-1 -> C * V-1 * cm-1

n = 1e19
m = 1.
E = KB * T / _q
t = _eps_0 / _q / n / m
d = np.sqrt(_eps_0 * E / (_q * n))
v = d / t
J = KB * T * n * m / d
U = n * E * m / (d**2)

units = {
    "grid": d,
    "eps": 1.,
    "Chi": E,
    "Eg": E,
    "Nc": n,
    "Nv": n,
    "mn": m,
    "mp": m,
    "Ndop": n,
    "Et": E,
    "tn": t,
    "tp": t,
    "Br": 1 / (t * n),
    "Cn": 1 / (t * n**2),
    "Cp": 1 / (t * n**2),
    "A": 1 / d,
    "G": U,
    "Snl": v,
    "Snr": v,
    "Spl": v,
    "Spr": v,
    "Lambda": 1.,
    "P_in": 1.
}
