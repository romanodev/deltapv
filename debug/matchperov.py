import jaxpv
import sesame
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt

L_ETM = 5e-5
L_Perov = 1.1e-4
L_HTM = 5e-5
A = 2e4

# JAXPV

region_ETM = lambda grid_ses: grid_ses <= L_ETM
region_Perov = lambda grid_ses: np.logical_and(L_ETM < grid_ses, grid_ses <= L_ETM + L_Perov)
region_HTM = lambda grid_ses: L_ETM + L_Perov < grid_ses

Perov = jaxpv.materials.create_material(Eg=1.5,
                                        Chi=3.9,
                                        eps=10,
                                        Nc=3.9e18,
                                        Nv=2.7e18,
                                        mn=2,
                                        mp=2,
                                        tn=1e-6,
                                        tp=1e-6,
                                        # Br=2.3e-9,
                                        A=A)
ETM = jaxpv.materials.create_material(Eg=4,
                                      Chi=4.1,
                                      eps=8.4,
                                      Nc=5.6e18,
                                      Nv=1.1e18,
                                      mn=191.4,
                                      mp=5.4,
                                      tn=1e-6,
                                      tp=1e-6,
                                      A=A)
HTM = jaxpv.materials.create_material(Eg=3.3,
                                      Chi=2.1,
                                      eps=20,
                                      Nc=2.2e19,
                                      Nv=1e18,
                                      mn=4.5,
                                      mp=361,
                                      tn=1e-6,
                                      tp=1e-6,
                                      A=A)


grid = np.linspace(0, L_ETM + L_Perov + L_HTM, 500)
des = jaxpv.simulator.create_design(grid)

des = jaxpv.simulator.add_material(des, ETM, region_ETM)
des = jaxpv.simulator.add_material(des, Perov, region_Perov)
des = jaxpv.simulator.add_material(des, HTM, region_HTM)

des = jaxpv.simulator.doping(des, 6e17, region_ETM)
des = jaxpv.simulator.doping(des, -1e18, region_HTM)

des = jaxpv.simulator.contacts(des, 1e7, 1e7, 1e7, 1e7)

ls = jaxpv.simulator.incident_light()

results = jaxpv.simulator.simulate(des, ls)

# SESAME

G = results["cell"].G * jaxpv.scales.gratedens
voltages, j_jaxpv = results["iv"]

grid = onp.linspace(0, L_ETM + L_Perov + L_HTM, 500)
sys = sesame.Builder(grid)

region_ETM = lambda grid_ses: grid_ses <= L_ETM
region_Perov = lambda grid_ses: onp.logical_and(L_ETM < grid_ses, grid_ses <= L_ETM + L_Perov)
region_HTM = lambda grid_ses: L_ETM + L_Perov < grid_ses

Perov = {
    "Eg": 1.5,
    "affinity": 3.9,
    "epsilon": 10,
    "Nc": 3.9e18,
    "Nv": 2.7e18,
    "mu_e": 2,
    "mu_h": 2,
    "tau_e": 1e-6,
    "tau_h": 1e-6,
    # "B": 2.3e-9,
    "Et": 0
}
ETM = {
    "Eg": 4,
    "affinity": 4.1,
    "epsilon": 8.4,
    "Nc": 5.6e18,
    "Nv": 1.1e18,
    "mu_e": 191.4,
    "mu_h": 5.4,
    "tau_e": 1e-6,
    "tau_h": 1e-6,
    "Et": 0
}
HTM = {
    "Eg": 3.3,
    "affinity": 2.1,
    "epsilon": 20,
    "Nc": 2.2e19,
    "Nv": 1e18,
    "mu_e": 4.5,
    "mu_h": 361,
    "tau_e": 1e-6,
    "tau_h": 1e-6,
    "Et": 0
}

sys.add_material(ETM, region_ETM)
sys.add_material(Perov, region_Perov)
sys.add_material(HTM, region_HTM)

sys.add_donor(6e17, region_ETM)
sys.add_acceptor(1e18, region_HTM)

sys.contact_type("Ohmic", "Ohmic")

sys.contact_S(1e7, 1e7, 1e7, 1e7)

def gfcn(x):
    return onp.interp(x, grid, G)

sys.generation(gfcn)

j_ses = sesame.IVcurve(sys, voltages, "outputs/1dhomo_V") * sys.scaling.current

plt.plot(voltages, j_jaxpv, marker="1", label="JAXPV")
plt.plot(voltages, j_ses, marker="2", label="SESAME")
plt.show()
