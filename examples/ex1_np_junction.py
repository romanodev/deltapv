import deltapv
import jax.numpy as np
import argparse
import matplotlib.pyplot as plt

L = 3e-4
grid = np.linspace(0, L, 500)
des = deltapv.simulator.create_design(grid)
material = deltapv.materials.create_material(Chi=3.9,
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
des = deltapv.simulator.add_material(des, material, lambda x: True)
des = deltapv.simulator.contacts(des, 1e7, 0, 0, 1e7)
des = deltapv.simulator.single_pn_junction(des, 1e17, -1e15, 5e-6)

ls = deltapv.simulator.incident_light()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    results = deltapv.simulator.simulate(des, ls)
    
    deltapv.plotting.plot_iv_curve(*results["iv"])
    deltapv.plotting.plot_bars(des)
    deltapv.plotting.plot_band_diagram(des, results["eq"], eq=True)
    deltapv.plotting.plot_charge(des, results["eq"])
    eff = results["eff"] * 100
