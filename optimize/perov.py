import jaxpv
from jax import numpy as np
import matplotlib.pyplot as plt

L_ETM = 5e-5
L_Perov = 1.1e-4
L_HTM = 5e-5
A = 2e4

region_ETM = lambda x: x <= L_ETM
region_Perov = lambda x: np.logical_and(L_ETM < x, x <= L_ETM + L_Perov)
region_HTM = lambda x: L_ETM + L_Perov < x

Perov = jaxpv.materials.create_material(Eg=1.5,
                                        Chi=3.9,
                                        eps=10,
                                        Nc=3.9e18,
                                        Nv=2.7e18,
                                        mn=2,
                                        mp=2,
                                        Br=2.3e-9,
                                        A=A)
ETM = jaxpv.materials.create_material(Eg=4,
                                      Chi=4.1,
                                      eps=8.4,
                                      Nc=5.6e18,
                                      Nv=1.1e18,
                                      mn=191.4,
                                      mp=5.4,
                                      A=A)
HTM = jaxpv.materials.create_material(Eg=3.3,
                                      Chi=2.1,
                                      eps=20,
                                      Nc=2.2e19,
                                      Nv=1e18,
                                      mn=4.5,
                                      mp=361,
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

if __name__ == "__main__":
    jaxpv.plotting.plot_bars(des)
    results = jaxpv.simulator.simulate(des, ls)
    jaxpv.plotting.plot_band_diagram(des, results["eq"], eq=True)
    jaxpv.plotting.plot_band_diagram(des, results["Voc"])
    jaxpv.plotting.plot_iv_curve(*results["iv"])
    plt.plot(des.grid, results["cell"].G * jaxpv.scales.gratedens)
    plt.show()
    print(results["eff"] * 100)
