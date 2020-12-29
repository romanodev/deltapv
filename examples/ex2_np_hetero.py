import jaxpv
from jax import numpy as np, ops
import matplotlib.pyplot as plt
import argparse

t1 = 25 * 1e-7
t2 = 4 * 1e-4
dd = 1e-7

grid = np.concatenate(
    (np.linspace(0, dd, 10,
                 endpoint=False), np.linspace(dd, t1 - dd, 50, endpoint=False),
     np.linspace(t1 - dd, t1 + dd, 10, endpoint=False),
     np.linspace(t1 + dd, (t1 + t2) - dd, 100,
                 endpoint=False), np.linspace((t1 + t2) - dd, (t1 + t2), 10)))

cell = jaxpv.simulator.create_design(grid)

CdS = jaxpv.materials.create_material(Nc=2.2e18,
                                      Nv=1.8e19,
                                      Eg=2.4,
                                      eps=10,
                                      Et=0,
                                      mn=100,
                                      mp=25,
                                      tn=1e-8,
                                      tp=1e-13,
                                      Chi=4.,
                                      A=1e4)

CdTe = jaxpv.materials.create_material(Nc=8e17,
                                       Nv=1.8e19,
                                       Eg=1.5,
                                       eps=9.4,
                                       Et=0,
                                       mn=320,
                                       mp=40,
                                       tn=5e-9,
                                       tp=5e-9,
                                       Chi=3.9,
                                       A=1e4)

cell = jaxpv.simulator.add_material(cell, CdS, lambda x: x < t1)
cell = jaxpv.simulator.add_material(cell, CdTe, lambda x: x >= t1)
cell = jaxpv.simulator.single_pn_junction(cell, 1e17, -1e15, t1)
cell = jaxpv.simulator.contacts(cell, 1.16e7, 1.16e7, 1.16e7, 1.16e7)

ls = jaxpv.simulator.incident_light()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save")
    args = parser.parse_args()

    voltages, j = jaxpv.simulator.iv_curve(cell, ls)

    plt.plot(voltages, j, "-o")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A/cm^2]")
    plt.title("pn-heterojunction example")
    if args.save is not None:
        plt.savefig(args.save)
    plt.show()
