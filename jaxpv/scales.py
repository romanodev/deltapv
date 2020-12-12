import scipy.constants as const
import math

##### Units re_quired in inputs #####
## grid : cm
## eps : unitless
## Chi , Eg , Et : eV
## Nc , Nv , Ndop : cm^(-3)
## mn , mp : cm^2 / V / s
## tn , tp : s
## Snl , Snr , Spl , Spr : cm / s
## G : cm^(-3) / s

T = 300.
KB = const.k
hc = const.c * const.h * 1e9  # J.nm
_q, _, _ = const.physical_constants['elementary charge']
_eps_0 = const.epsilon_0 * 1e-2  # C * V-1 * m-1 -> C * V-1 * cm-1


n = 1e19
m = 1
E = KB * T / _q
t = _eps_0 / _q / n / m
d = math.sqrt(_eps_0 * E / (_q * n))
v = d / t
J = KB * T * n * m / d
U = n * E * m / (d**2)
