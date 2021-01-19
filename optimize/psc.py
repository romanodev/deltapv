import os
os.environ["LOGLEVEL"] = "INFO"
import jaxpv
from jax import numpy as np, value_and_grad
import matplotlib.pyplot as plt

L_ETM = 5e-5
L_Perov = 1.1e-4
L_HTM = 5e-5
N = 500
A = 2e4
tau = 1e-6
S = 1e7

Perov = jaxpv.materials.create_material(Eg=1.5,
                                        Chi=3.9,
                                        eps=10,
                                        Nc=3.9e18,
                                        Nv=2.7e18,
                                        mn=2,
                                        mp=2,
                                        tn=tau,
                                        tp=tau,
                                        Br=2.3e-9,
                                        A=A)

region_ETM = lambda x: x <= L_ETM
region_Perov = lambda x: np.logical_and(L_ETM < x, x <= L_ETM + L_Perov)
region_HTM = lambda x: L_ETM + L_Perov < x


def psceff(params):

    Eg_ETM = params[0]
    Chi_ETM = params[1]
    eps_ETM = params[2]
    Nc_ETM = 10**params[3]
    Nv_ETM = 10**params[4]
    mn_ETM = params[5]
    mp_ETM = params[6]
    Eg_HTM = params[7]
    Chi_HTM = params[8]
    eps_HTM = params[9]
    Nc_HTM = 10**params[10]
    Nv_HTM = 10**params[11]
    mn_HTM = params[12]
    mp_HTM = params[13]
    Nd_ETM = 10**params[14]
    Na_HTM = 10**params[15]

    ETM = jaxpv.materials.create_material(Eg=Eg_ETM,
                                          Chi=Chi_ETM,
                                          eps=eps_ETM,
                                          Nc=Nc_ETM,
                                          Nv=Nv_ETM,
                                          mn=mn_ETM,
                                          mp=mp_ETM,
                                          tn=tau,
                                          tp=tau,
                                          A=A)
    HTM = jaxpv.materials.create_material(Eg=Eg_HTM,
                                          Chi=Chi_HTM,
                                          eps=eps_HTM,
                                          Nc=Nc_HTM,
                                          Nv=Nv_HTM,
                                          mn=mn_HTM,
                                          mp=mp_HTM,
                                          tn=tau,
                                          tp=tau,
                                          A=A)

    grid = np.linspace(0, L_ETM + L_Perov + L_HTM, N)
    des = jaxpv.simulator.create_design(grid)

    des = jaxpv.simulator.add_material(des, ETM, region_ETM)
    des = jaxpv.simulator.add_material(des, Perov, region_Perov)
    des = jaxpv.simulator.add_material(des, HTM, region_HTM)
    des = jaxpv.simulator.doping(des, Nd_ETM, region_ETM)
    des = jaxpv.simulator.doping(des, -Na_HTM, region_HTM)
    des = jaxpv.simulator.contacts(des, S, S, S, S)

    ls = jaxpv.simulator.incident_light()

    results = jaxpv.simulator.simulate(des, ls)
    eff = results["eff"] * 100

    return eff


if __name__ == "__main__":

    x = np.array([
        4,
        4.1,
        8.4,
        18.8,
        18,
        191.4,
        5.4,  # ETM
        3.3,
        2.1,
        20,
        19.3,
        18,
        4.5,
        361,  # HTM
        17.8,
        18  # doping
    ])

    pscvg = value_and_grad(psceff)
    eff, grad = pscvg(x)

    print(x)
    print(eff)
    print(grad)
