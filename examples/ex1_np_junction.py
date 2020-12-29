import jaxpv
import jax.numpy as np
import argparse
import matplotlib.pyplot as plt

L = 3e-4
grid = np.linspace(0, L, 500)
cell = jaxpv.simulator.create_cell(grid)
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
                                           A=1e4)
cell = jaxpv.simulator.add_material(cell, material, lambda x: True)
cell = jaxpv.simulator.contacts(cell, 1e7, 0, 0, 1e7)
cell = jaxpv.simulator.single_pn_junction(cell, 1e17, -1e15, 50e-7)

phi = 1e17  # photon flux [cm-2 s-1)]
alpha = 2.3e4  # absorption coefficient [cm-1]
G = phi * alpha * np.exp(-alpha * grid)  # cm-3 s-1
cell = jaxpv.simulator.custom_generation(cell, G)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    voltages, j = jaxpv.simulator.iv_curve(cell)

    plt.plot(voltages, j, "-o")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A/cm^2]")
    plt.title("pn-junction")
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()
