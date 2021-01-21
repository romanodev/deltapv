import jaxpv
import jax.numpy as np
import argparse
import matplotlib.pyplot as plt

L = 3e-4
grid = np.linspace(0, L, 500)
des = jaxpv.simulator.create_design(grid)
material = jaxpv.materials.create_material(Chi=3.9,
                                           Eg=1.5,
                                           eps=9.4,
                                           Nc=8e17,
                                           Nv=1.8e19,
                                           mn=100,
                                           mp=100,
                                           Et=0,
                                           tn=1e-8,
                                           tp=1e-8,
                                           A=2e4)
des = jaxpv.simulator.add_material(des, material, lambda x: True)
des = jaxpv.simulator.contacts(des, 1e7, 0, 0, 1e7)
des = jaxpv.simulator.single_pn_junction(des, 1e17, -1e15, 50e-7)

ls = jaxpv.simulator.incident_light()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    results = jaxpv.simulator.simulate(des, ls)
    v, j = results["iv"]

    jaxpv.plotting.plot_iv_curve(v, j)
    jaxpv.plotting.plot_bars(des)
    jaxpv.plotting.plot_band_diagram(des, results["eq"], eq=True)
    jaxpv.plotting.plot_band_diagram(des, results["Voc"])
    jaxpv.plotting.plot_charge(des, results["eq"])
    eff = results["eff"] * 100
    print(f"efficiency: {eff}%")
