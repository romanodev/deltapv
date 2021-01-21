from scipy.constants import *

T = 300
meff_n = 1.08
meff_p = 0.81
Nc = 2 * (2 * pi * k * T * meff_n * m_e / h ** 2)**1.5 / 1e6  # / cm^3
Nv = 2 * (2 * pi * k * T * meff_p * m_e / h ** 2)**1.5 / 1e6  # / cm^3

print(Nc)
print(Nv)
