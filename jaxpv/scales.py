from scipy import constants as const
from jax import numpy as np

# Units required in inputs:
# grid : cm
# eps : unitless
# Chi , Eg , Et : eV
# Nc , Nv , Ndop : cm^(-3)
# mn , mp : cm^2 / (V s)
# tn , tp : s
# Snl , Snr , Spl , Spr : cm / s
# G : cm^(-3) / s

# Physical constants:
kB = const.k  # J / K
hc = const.c * const.h * 1e9  # J m -> J nm
q = const.e  # C
eps0 = const.epsilon_0 * 1e-2  # C / (V m) -> C / (V cm)

# Constants for scaling physical quantities to be dimensionless:
density = 1e19  # 1 / cm^3
mobility = 1.  # cm^2 / (V s)
temperature = 300.  # K
energy = kB * temperature / q  # J -> eV through dividing by numerical value of q
time = eps0 / q / density / mobility  # s
length = np.sqrt(eps0 * kB * temperature / (q**2 * density))  # cm
velocity = length / time  # cm / s
current = kB * temperature * density * mobility / length  # A / cm^2
gratedens = density * mobility * kB * temperature / (q * length**2)  # 1 / (cm^3 s)

# Scaling dictionary for input quantities:
units = {
    "grid": length,
    "eps": 1., # relative permittivity dimensionless
    "Chi": energy,
    "Eg": energy,
    "Nc": density,
    "Nv": density,
    "mn": mobility,
    "mp": mobility,
    "Ndop": density,
    "Et": energy,
    "tn": time,
    "tp": time,
    "Br": 1 / (time * density),
    "Cn": 1 / (time * density**2),
    "Cp": 1 / (time * density**2),
    "A": 1 / length,
    "alpha": 1 / length,
    "G": gratedens,
    "Snl": velocity,
    "Snr": velocity,
    "Spl": velocity,
    "Spr": velocity,
    "Lambda": 1.,  # perform optical computations with original units
    "P_in": 1.  # perform optical computations with original units
}
