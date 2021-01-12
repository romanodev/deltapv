import os
os.environ["LOGLEVEL"] = "WARNING"
import jaxpv
from jax import numpy as np, value_and_grad
import argparse
import matplotlib.pyplot as plt

ETLx = 2e-6
HTLx = 2e-6
dd = 1e-7

# https://pubs.rsc.org/lv/content/articlehtml/2019/ee/c8ee01576g
NiO = jaxpv.materials.create_material(Eg=2.8,
                                      Chi=2.5,
                                      eps=8,
                                      mn=1e-5,
                                      mp=1,
                                      Nc=1e19,
                                      Nv=1e19,
                                      tn=1e-9,
                                      tp=1e-9,
                                      Et=0,
                                      A=2e4)
Perov = jaxpv.materials.create_material(Eg=1.78,
                                        Chi=4,
                                        eps=20,
                                        mn=5,
                                        mp=5,
                                        Nc=1e19,
                                        Nv=1e19,
                                        tn=1e-9,
                                        tp=1e-9,
                                        Et=0,
                                        A=2e4)
PCBM = jaxpv.materials.create_material(Eg=1.8,
                                       Chi=5.9,
                                       eps=3.9,
                                       mn=1,
                                       mp=1,
                                       Nc=1e19,
                                       Nv=1e19,
                                       tn=1e-9,
                                       tp=1e-9,
                                       Et=0,
                                       A=2e4)
Si = jaxpv.materials.load_material("Si")



def calc_eff(PEROVx=2e-5):

    grid = np.concatenate([
        np.linspace(0, ETLx - dd, 100, endpoint=False),
        np.linspace(ETLx - dd, ETLx + dd, 200, endpoint=False),
        np.linspace(ETLx + dd, ETLx + PEROVx - dd, 100, endpoint=False),
        np.linspace(ETLx + PEROVx - dd, ETLx + PEROVx + dd, 200, endpoint=False),
        np.linspace(ETLx + PEROVx + dd, ETLx + PEROVx + HTLx, 100)
    ])
    des = jaxpv.simulator.create_design(grid)

    des = jaxpv.simulator.add_material(des, Si, lambda x: x <= ETLx)
    des = jaxpv.simulator.add_material(
        des, Perov, lambda x: np.logical_and(ETLx < x, x <= ETLx + PEROVx))
    des = jaxpv.simulator.add_material(des, Si, lambda x: ETLx + PEROVx < x)

    des = jaxpv.simulator.contacts(des, 1e7, 0, 0, 1e7)
    des = jaxpv.simulator.doping(des, 1e18, lambda x: x <= ETLx)
    des = jaxpv.simulator.doping(des, -1e18, lambda x: ETLx + PEROVx < x)

    ls = jaxpv.simulator.incident_light()

    results = jaxpv.simulator.simulate(des, ls)

    return results["eff"]


if __name__ == "__main__":

    f = value_and_grad(calc_eff)
    x = 2e-5
    for _ in range(5):
        y, dydl = f(x)
        print(y, dydl)
        x += 1e-5 * dydl
