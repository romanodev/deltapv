import sesame
import numpy as np
import matplotlib.pyplot as plt

L = 1e-4
JPOS = 5e-5
N = 500
S = 1e7
grid = np.linspace(0, L, N)
Eg_0 = 1.12
MAT = {
    "epsilon": 11.7,
    "affinity": 4.05,
    "Eg": Eg_0,
    "Nc": 10**19.5051499783,
    "Nv": 10**19.2552725051,
    "mu_e": 10**3.1461280357,
    "mu_h": 10**2.6532125138,
    "tau_e": 10**(-8.0),
    "tau_h": 10**(-8.0),
    "Et": 0
}

sim = sesame.Builder(grid)
sim.add_material(MAT)

n_region = lambda x: x < JPOS
p_region = lambda x: x >= JPOS

sim.add_donor(1e18, n_region)
sim.add_acceptor(1e18, p_region)

sim.contact_type("Ohmic", "Ohmic")
sim.contact_S(S, S, S, S)

# G = np.load("debug/G_flat.npy")
G = np.load("debug/G_ends.npy")
gfunc = lambda x: np.interp(x, grid, G)
sim.generation(gfunc)

voltages = 0.05 * np.arange(16)

sim.Eg[0] = 5 / sim.scaling.energy
sim.Eg[-1] = 5 / sim.scaling.energy

j = sesame.IVcurve(sim, voltages, "debug/sesame/gends")
j = j * sim.scaling.current

print(1e2 * np.max(voltages * j * 1e4) / 1e3)
plt.plot(voltages, j)
plt.show()
